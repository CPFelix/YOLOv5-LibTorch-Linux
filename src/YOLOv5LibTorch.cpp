#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <algorithm>
#include <iostream>
#include <time.h>


std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5)
{
        std::vector<torch::Tensor> output;
        for (size_t i=0; i < preds.sizes()[0]; ++i)
        {
            torch::Tensor pred = preds.select(0, i);
            pred = pred.to(at::kCPU);
            // Filter by scores
            torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
            pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
            if (pred.sizes()[0] == 0) continue;

            // (center_x, center_y, w, h) to (left, top, right, bottom)
            pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
            pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
            pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
            pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

            // Computing scores and classes
            std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
            pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
            pred.select(1, 5) = std::get<1>(max_tuple);

            torch::Tensor  dets = pred.slice(1, 0, 6);

            torch::Tensor keep = torch::empty({dets.sizes()[0]});
            torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
            std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
            torch::Tensor v = std::get<0>(indexes_tuple);
            torch::Tensor indexes = std::get<1>(indexes_tuple);
            int count = 0;
            while (indexes.sizes()[0] > 0)
            {
                keep[count] = (indexes[0].item().toInt());
                count += 1;

                // Computing overlaps
                torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
                for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
                {
                    lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                    tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                    rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                    bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                    widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                    heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
                }
                torch::Tensor overlaps = widths * heights;

                // FIlter by IOUs
                torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
                indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
            }
            keep = keep.toType(torch::kInt64);
            output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
        }
        return output;
}


int main()
{
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    torch::DeviceType device_type = at::kCPU; // 定义设备类型
    if (torch::cuda::is_available())
	device_type = at::kCUDA;
    // Loading  Module
    torch::jit::script::Module module = torch::jit::load("../best.torchscript.pth");
    module.to(device_type);
    std::vector<std::string> classnames;
    std::ifstream f("../truck.names");
    std::string name = "";
    while (std::getline(f, name))
    {
	classnames.push_back(name);
    }

    //cv:: VideoCapture cap = cv::VideoCapture(0);
    //cap.set(cv::CV_CAP_PROP_FRAME_WIDTH, 1920);
    //cap.set(cv::CV_CAP_PROP_FRAME_HEIGHT, 1080);
    cv::Mat img;
    cv::Mat frame = cv::imread("../data/car_plate_public_roads_avenue_none_train_p_day_20210329_2.jpg");
    int count = 1;
    while(count == 1)
    {
	count++;
        clock_t start = clock();
	//cap.read(frame);
        if(frame.empty())
        {
           std::cout << "Read frame failed!" << std::endl;
           break;
        }
	int netw = 320;
	int neth = 320;
	int stride = 32;
	int imgW = frame.cols;
	int imgH = frame.rows;
	// Preparing input tensor
	//cv::resize(frame, img, cv::Size(640, 640));
	//pad操作
	float ratioWidth, ratioHeight, ratio;
	ratioWidth = (float)netw / imgW;
	ratioHeight = (float)neth / imgH;
	ratio = ratioWidth;
	if (ratioWidth > ratioHeight)
		ratio = ratioHeight;
	int tmpWidth = int(imgW * ratio);
	int tmpHeight = int(imgH * ratio);
	int input_w;
	int input_h;
	if ((tmpWidth % stride) != 0)
	{
		input_w = stride * int(tmpWidth / stride + 1);
	}
	else
	{
		input_w = tmpWidth;
	}
	if ((tmpHeight % stride) != 0)
	{
		input_h = stride * int(tmpHeight / stride + 1);
	}
	else
	{
		input_h = tmpHeight;
	}
	input_w = netw;
	input_h = neth;	        

	cv::Mat img2;
	cv::resize(frame, img2, cv::Size(tmpWidth, tmpHeight));
	cv::Mat temp(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	cv::Mat roi = temp(cv::Rect((input_w - tmpWidth) / 2, (input_h - tmpHeight) / 2, img2.cols, img2.rows));
	img2.copyTo(roi);
	img = temp;

        // Preparing input tensor
        //cv::resize(frame, img, cv::Size(320, 320));
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  // BGR -> RGB
	img.convertTo(img, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
	auto imgTensor = torch::from_blob(img.data, { 1, img.rows, img.cols, img.channels() }).to(device_type);
	imgTensor = imgTensor.permute({ 0, 3, 1, 2 }).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)
	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(imgTensor);
	torch::jit::IValue output = module.forward(inputs);
	auto preds = output.toTuple()->elements()[0].toTensor();
	std::vector<torch::Tensor> dets = non_max_suppression(preds, 0.5, 0.45); 
        std::cout << "dets.size():" << dets.size() << std::endl;
        if (dets.size() > 0)
        {
            // Visualize result
            for (size_t i=0; i < dets[0].sizes()[0]; ++ i)
            {
                float left = (dets[0][i][0].item().toFloat() - (input_w - tmpWidth) / 2)  / ratio;
                float top = (dets[0][i][1].item().toFloat() - (input_h - tmpHeight) / 2) / ratio;
                float right = (dets[0][i][2].item().toFloat() - (input_w - tmpWidth) / 2) / ratio;
                float bottom = (dets[0][i][3].item().toFloat() - (input_h - tmpHeight) / 2) / ratio;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();

		cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 0, 255), 2);
		cv::putText(frame, classnames[classID] + ": " + cv::format("%.2f", score), cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(0, 0, 255), 2);
            }
        }
        //cv::putText(frame, "FPS: " + std::to_string(int(1e7 / (clock() - start))), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("", frame);
	cv::waitKey(0);
        //if(cv::waitKey(1)== 2700) break;
    }
    return 0;
}
