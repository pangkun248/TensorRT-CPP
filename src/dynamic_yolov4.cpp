#include "yolov4.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "common.h"
#include "NvInfer.h"

using namespace std;
using namespace nvinfer1;
using namespace cv;

int net_c = 3;
float obj_threshold = 0.5;
float nms_threshold = 0.5;
int cls_num = 18;

vector<string> labels = { "WhitehairedBanshee","UndeadSkeleton","WhitehairedMonster","SlurryMonster", "MiniZalu",
                          "Dopelliwin","ShieldAxe", "SkeletonKnight", "Zalu", "Cyclone", "SlurryBeggar", "Gerozaru",
                          "Catalog","InfectedMonst", "Gold", "StormRider", "Close", "Door",};
int anchors[18] = { 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401 };
typedef struct DetectionRes {
    float x1,y1,x2,y2,x,y,w,h;
    float prob,score,max_cls_ind;
} DetectionRes;


inline u_int32_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}


inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

void DoNms(vector<DetectionRes>& detections, float nmsThresh){
    auto iouCompute = [](DetectionRes &lbox, DetectionRes &rbox) {
        float interBox[] = {
                max(lbox.x1, rbox.x1), //x1
                max(lbox.y1, rbox.y1), //y1
                min(lbox.x2, rbox.x2), //x2
                min(lbox.y2, rbox.y2), //y2
        };

        if (interBox[1] >= interBox[3] || interBox[0] >= interBox[2])
            return 0.0f;

        float interBoxS = (interBox[2] - interBox[0] + 1) * (interBox[3] - interBox[1] + 1);
        float lbox_area = (lbox.x2-lbox.x1+1)*(lbox.y2-lbox.y1+1);
        float rbox_area = (rbox.x2-rbox.x1+1)*(rbox.y2-rbox.y1+1);
        float extra = 0.00001;

        return interBoxS / (lbox_area+rbox_area-interBoxS+extra);
    };

    vector<DetectionRes> result;
    for (unsigned int m = 0; m < detections.size(); ++m) {
        result.push_back(detections[m]);
        for (unsigned int n = m + 1; n < detections.size(); ++n) {
            if (iouCompute(detections[m], detections[n]) > nmsThresh) {
                detections.erase(detections.begin() + n);
                --n;
            }
        }
    }
    detections = move(result);
}
// 这里要把TRT引擎初始化拿出来.否则每次都会重新生成
IRuntime* runtime = createInferRuntime(gLogger);
// 读取TRT模型文件
std::string load_trt(){
    std::string cached_path = "/home/cmv/PycharmProjects/YOLOv4-PyTorch/deploy/yolov4_-1_3_-1_-1_INT8_sigmoid.trt";;
    std::ifstream fin(cached_path);
    std::string temp_engine = "";
    while (fin.peek() != EOF){
        std::stringstream buffer;
        buffer << fin.rdbuf();
        temp_engine.append(buffer.str());
    }
    fin.close();
    return temp_engine;
}
std::string cached_engine = load_trt();
// 反序列化
ICudaEngine* engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);

// 定义inference
IExecutionContext *context = engine->createExecutionContext();

// 获取输入输出层的索引 这里的input与output必须与 onnx模型中的一致  暂时用不到
//int inputIndex = engine->getBindingIndex("input");
//int outputIndex = engine->getBindingIndex("output");
// buffers 是指输入和输出的总数 显然 YOLOv3 YOLOv4 有三个yolo层 tiny有两个yolo层 所以v3=v4=4 tiny=3
const int target_bind_nb = 4;
void* buffers[target_bind_nb];

std::vector<int64_t> bufferSize;
int nbBindings = engine->getNbBindings();

vector<vector<DetectionRes> > dection(string img_path,int bs=1,int net_h=416,int net_w=416){
    // 读取图片
    cv::Mat img = cv::imread(img_path);
    int org_h = img.rows;
    int org_w = img.cols;
    // 设置模型输入尺寸 只有context可以setDim(即使修改了也不会在engine中体现)也可以getDim engine只有getDim方法(负责从TRT模型中读取)
    auto input_dims = nvinfer1::Dims4{bs, 3, net_h, net_w};
    context->setBindingDimensions(0, input_dims);
    // 准备输出数据 以及开辟显存
    float output1[bs][3*(cls_num+5)][net_h/8][net_w/8];
    float output2[bs][3*(cls_num+5)][net_h/16][net_w/16];
    float output3[bs][3*(cls_num+5)][net_h/32][net_w/32];
    bufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        nvinfer1::DataType data_type = engine->getBindingDataType(i);
        u_int32_t totalSize = volume(dims) * 1 * getElementSize(data_type);
        bufferSize[i] = totalSize;
        cudaMalloc(&buffers[i], totalSize);
    }
//    cv::Mat org_img = cv::imread(img_path);
    cv::Mat float_img;
    // 图片预处理
    Mat rgb;
    cv::cvtColor(img, rgb,  cv::COLOR_BGR2RGB);
    // resize
    Mat resized_img;
    cv::resize(rgb, resized_img, cv::Size(net_w, net_h));
    resized_img.convertTo(float_img, CV_32F, 1.0/255);
    // 图片数据转为inputs
    vector<Mat> input_channels(net_c);
    cv::split(float_img, input_channels);
    vector<float> result(net_h * net_w * net_c);
    auto data = result.data();
    int channelLength = net_h * net_w;
    for (int i = 0; i < net_c; ++i) {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
    vector<float> curInput = result;
    cudaMemcpy(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice);
    // 同步执行inference
    context->executeV2(buffers);
    cudaMemcpy(output1, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost);
    cudaMemcpy(output2, buffers[2], bufferSize[2], cudaMemcpyDeviceToHost);
    cudaMemcpy(output3, buffers[3], bufferSize[3], cudaMemcpyDeviceToHost);
    vector<vector<DetectionRes> > all_detections(cls_num);
    for (int i = 0; i < bs; ++i) { // bs 默认为1
        for (int a = 0; a < 3; ++a) { // 三个yolo层
            for (int h = 0; h < net_h/8; ++h) {
                for (int w = 0; w < net_w/8; ++w) {
                    float max_label = std::max({output1[i][a*23+5][h][w],output1[i][a*23+6][h][w],output1[i][a*23+7][h][w],output1[i][a*23+8][h][w],output1[i][a*23+9][h][w],output1[i][a*23+10][h][w],output1[i][a*23+11][h][w],output1[i][a*23+12][h][w],output1[i][a*23+13][h][w],output1[i][a*23+14][h][w],output1[i][a*23+15][h][w],output1[i][a*23+16][h][w],output1[i][a*23+17][h][w],output1[i][a*23+18][h][w],output1[i][a*23+19][h][w],output1[i][a*23+20][h][w],output1[i][a*23+21][h][w],output1[i][a*23+22][h][w]});
                    if (output1[i][a*23+4][h][w]*max_label< obj_threshold){
                        continue;
                    }
                    DetectionRes det;
                    for (int k = 0; k < cls_num; ++k) {
                        if (max_label == output1[i][a*23+5+k][h][w]){
                            det.max_cls_ind = k;
                            break;
                        }
                    }
                    det.prob = output1[i][a*23+4][h][w];
                    det.x = (w+ output1[i][a*23][h][w])*8;
                    det.y = (h+ output1[i][a*23+1][h][w])*8;
                    det.w = output1[i][a*23+2][h][w]/(1-output1[i][a*23+2][h][w])*anchors[2*a];
                    det.h = output1[i][a*23+3][h][w]/(1-output1[i][a*23+3][h][w])*anchors[2*a+1];
                    det.x1 = det.x -det.w/2;
                    det.y1 = det.y -det.h/2;
                    det.x2 = det.x +det.w/2;
                    det.y2 = det.y +det.h/2;
                    det.score = max_label*output1[i][a*23+4][h][w];
                    all_detections[det.max_cls_ind].push_back(det);
                }
            }
            for (int h = 0; h < net_h/16; ++h) {
                for (int w = 0; w < net_w/16; ++w) {
                    float max_label = std::max({output2[i][a*23+5][h][w],output2[i][a*23+6][h][w],output2[i][a*23+7][h][w],output2[i][a*23+8][h][w],output2[i][a*23+9][h][w],output2[i][a*23+10][h][w],output2[i][a*23+11][h][w],output2[i][a*23+12][h][w],output2[i][a*23+13][h][w],output2[i][a*23+14][h][w],output2[i][a*23+15][h][w],output2[i][a*23+16][h][w],output2[i][a*23+17][h][w],output2[i][a*23+18][h][w],output2[i][a*23+19][h][w],output2[i][a*23+20][h][w],output2[i][a*23+21][h][w],output2[i][a*23+22][h][w]});
                    if (output2[i][a*23+4][h][w]*max_label< obj_threshold){
                        continue;
                    }
                    DetectionRes det;
                    for (int k = 0; k < cls_num; ++k) {
                        if (max_label == output2[i][a*23+5+k][h][w]){
                            det.max_cls_ind = k;
                            break;
                        }
                    }
                    det.prob = output2[i][a*23+4][h][w];
                    det.x = (w+ output2[i][a*23][h][w])*16;
                    det.y = (h+ output2[i][a*23+1][h][w])*16;
                    det.w = output2[i][a*23+2][h][w]/(1-output2[i][a*23+2][h][w])*anchors[2*(a+3)];
                    det.h = output2[i][a*23+3][h][w]/(1-output2[i][a*23+3][h][w])*anchors[2*(a+3)+1];
                    det.x1 = det.x -det.w/2;
                    det.y1 = det.y -det.h/2;
                    det.x2 = det.x +det.w/2;
                    det.y2 = det.y +det.h/2;
                    det.score = max_label*output2[i][a*23+4][h][w];
                    all_detections[det.max_cls_ind].push_back(det);

                }
            }
            for (int h = 0; h < net_h/32; ++h) {
                for (int w = 0; w < net_w/32; ++w) {
                    float max_label = std::max({output3[i][a*23+5][h][w],output3[i][a*23+6][h][w],output3[i][a*23+7][h][w],output3[i][a*23+8][h][w],output3[i][a*23+9][h][w],output3[i][a*23+10][h][w],output3[i][a*23+11][h][w],output3[i][a*23+12][h][w],output3[i][a*23+13][h][w],output3[i][a*23+14][h][w],output3[i][a*23+15][h][w],output3[i][a*23+16][h][w],output3[i][a*23+17][h][w],output3[i][a*23+18][h][w],output3[i][a*23+19][h][w],output3[i][a*23+20][h][w],output3[i][a*23+21][h][w],output3[i][a*23+22][h][w]});
                    if (output3[i][a*23+4][h][w]*max_label< obj_threshold){
                        continue;
                    }
                    DetectionRes det;
                    for (int k = 0; k < cls_num; ++k) {
                        if (max_label == output3[i][a*23+5+k][h][w]){
                            det.max_cls_ind = k;
                            break;
                        }
                    }
                    det.prob = output3[i][a*23+4][h][w];
                    det.x = (w+ output3[i][a*23][h][w])*32;
                    det.y = (h+ output3[i][a*23+1][h][w])*32;
                    det.w = output3[i][a*23+2][h][w]/(1-output3[i][a*23+2][h][w])*anchors[2*(a+6)];
                    det.h = output3[i][a*23+3][h][w]/(1-output3[i][a*23+3][h][w])*anchors[2*(a+6)+1];
                    det.x1 = det.x -det.w/2;
                    det.y1 = det.y -det.h/2;
                    det.x2 = det.x +det.w/2;
                    det.y2 = det.y +det.h/2;
                    det.score = max_label*output3[i][a*23+4][h][w];
                    all_detections[det.max_cls_ind].push_back(det);

                }
            }

        }

    }
    // 每个类别排序，然后做一遍NMS
    for(auto &elem : all_detections){
        sort(elem.begin(), elem.end(), [=](const DetectionRes & left, const DetectionRes & right) {
            return left.score > right.score;
        });
        DoNms(elem,nms_threshold);
    }
    // 释放显存
    for (int i = 0; i < nbBindings; ++i){
        cudaFree(buffers[i]);
    }
//    for(int i=0;i<all_detections.size();i++){ // 第 i 个类别
//        for(int j=0;j<all_detections[i].size();j++){ // 第i个类别中第j个pred_box
//            // 为每个类别产生随机颜色
//            int red = rand() % 256;
//            int green = rand() % 256;
//            int blue = rand() % 256;
//
//            all_detections[i][j].x1=all_detections[i][j].x1/net_w*org_w;
//            all_detections[i][j].y1=all_detections[i][j].y1/net_h*org_h;
//            all_detections[i][j].x2=all_detections[i][j].x2/net_w*org_w;
//            all_detections[i][j].y2=all_detections[i][j].y2/net_h*org_h;
//            std::cout<< "class_name"<< " x1"<< " y1"<< " x2"<< " y2"<< " obj_score"<< " obj*cls_score"<<std::endl;
//            std::cout<<labels[i] << ", " << all_detections[i][j].x1 << ", " << all_detections[i][j].y1 << ", " << all_detections[i][j].x2<< ", "<< all_detections[i][j].y2<<", "<<all_detections[i][j].prob<<",  "<<all_detections[i][j].score<<std::endl;
//            cv::putText(org_img, labels[i], cv::Point(all_detections[i][j].x1, all_detections[i][j].y1), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(red, green, blue), 1,8);
//            cv::rectangle(org_img, cv::Point(all_detections[i][j].x1, all_detections[i][j].y1), cv::Point(all_detections[i][j].x2, all_detections[i][j].y2), cv::Scalar(red, green, blue), 1, 1, 0);
//        }
//    }
//    imshow("show_window",org_img);
//    cv::waitKey(0);
    return all_detections;
}
