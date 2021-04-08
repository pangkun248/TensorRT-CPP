#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
typedef struct DetectionRes {
    float x1,y1,x2,y2;
    float prob,score,max_cls_ind;
} DetectionRes;
vector<string> labels = { "WhitehairedBanshee","UndeadSkeleton","WhitehairedMonster","SlurryMonster", "MiniZalu",
                          "Dopelliwin","ShieldAxe", "SkeletonKnight", "Zalu", "Cyclone", "SlurryBeggar", "Gerozaru",
                          "Catalog","InfectedMonst", "Gold", "StormRider", "Close", "Door",};
vector<vector<DetectionRes>> dection(std::string path);
int main() {
    std::string img_path;
    while (1) {
        cout << "输入一个图片路径: ";
        cin >> img_path;
        ifstream fin(img_path);
        if (!fin) {
            std::cout << "这不是一个文件!" << endl;
            continue;
        }
        auto t_start = std::chrono::high_resolution_clock::now();
//        cv::Mat org_img = cv::imread(img_path);
        vector<vector<DetectionRes>> all_detections = dection(img_path);
//        auto org_h = org_img.rows;
//        auto org_w = org_img.cols;
//        int net_h = 416;
//        int net_w = 416;
//        for(int i=0;i<all_detections.size();i++){ // 第 i 个类别
//            for(int j=0;j<all_detections[i].size();j++){ // 第i个类别中第j个pred_box
//                // 为每个类别产生随机颜色
//                int red = rand() % 256;
//                int green = rand() % 256;
//                int blue = rand() % 256;
//
//                all_detections[i][j].x1=all_detections[i][j].x1/net_w*org_w;
//                all_detections[i][j].y1=all_detections[i][j].y1/net_h*org_h;
//                all_detections[i][j].x2=all_detections[i][j].x2/net_w*org_w;
//                all_detections[i][j].y2=all_detections[i][j].y2/net_h*org_h;
//                std::cout<< "class_name"<< " x1"<< " y1"<< " x2"<< " y2"<< " obj_score"<< " obj*cls_score"<<std::endl;
//                std::cout<<labels[i] << ", " << all_detections[i][j].x1 << ", " << all_detections[i][j].y1 << ", " << all_detections[i][j].x2<< ", "<< all_detections[i][j].y2<<", "<<all_detections[i][j].prob<<",  "<<all_detections[i][j].score<<std::endl;
//                cv::putText(org_img, labels[i], cv::Point(all_detections[i][j].x1, all_detections[i][j].y1), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(red, green, blue), 1,8);
//                cv::rectangle(org_img, cv::Point(all_detections[i][j].x1, all_detections[i][j].y1), cv::Point(all_detections[i][j].x2, all_detections[i][j].y2), cv::Scalar(red, green, blue), 1, 1, 0);
//        }
//    }
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = (std::chrono::duration<float, std::milli>(t_end - t_start).count());
    std::cout << "Inference take: " << total << " ms." << endl;
//    imshow("show_window",org_img);
//    cv::waitKey(0);
    }
}

// /home/cmv/PycharmProjects/YOLOv4-PyTorch/data/wenyi/JPGImages/000352.jpg