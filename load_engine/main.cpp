#define METHOD 1  // 0: camera 1: video 2: image
#define REMOTE 1  // 0: not remote operation (by ssh), 1: remote operation (by ssh)
#define PLATFORM 0  // 0: X86 PC, 1: Nvidia Jetson
#define HAND_TRACKING 1 // 0: do not enable hungarian-based hand tracking, 1: enable

#include "InferenceResult.h"
#include "YOLO.hpp"
#include "Skeleton.hpp"
#include "Gesture.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <chrono>
// #include <queue>  // included in Net.hpp

// #include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

// #include "NvInfer.h"  // included in Net.hpp
// #include "argsParser.h"  // included in Net.hpp
// #include "buffers.h"  // included in Net.hpp
// #include "common.h"  // included in Net.hpp
#include "logger.h"
// #include "parserOnnxConfig.h"
// #include "sampleEngines.h"  // included in Net.hpp
// #include "util/npp_common.h"
// #include "util/process.h"

const std::string SampleName = "TensorRT.DynamicGesture";

/**
 * @brief 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
void printHelpInfo() {
  std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or "
               "--datadir=<path to data directory>] [--useDLACore=<int>] [--preprocess=<int>]"
            << std::endl;
  std::cout << "--help          Display help information" << std::endl;
  std::cout << "--datadir       Specify path to a data directory, overriding "
               "the default. This option can be used "
               "multiple times to add multiple directories. If no data "
               "directories are given, the default is to use "
               "(data/samples/mnist/, data/mnist/)"
            << std::endl;
  std::cout << "--useDLACore=N  Specify a DLA engine for layers that support "
               "DLA. Value can range from 0 to n-1, "
               "where n is the number of DLA engines on the platform."
            << std::endl;
  std::cout << "--preprocess=N or -p=N  "
               "指定前处理所有设备，-1表示使用测试数据，0表示使用CPU做前处理，1表示使用GPU做前处理"
            << std::endl;
  std::cout << "--int8          Run in Int8 mode." << std::endl;
  std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

/**
 * @brief Create a Dummy Skeleton Input object
 * 
 * @return std::vector<float> 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
std::vector<float> createDummySkeletonInput() {
    std::vector<float> skeleton{
      0.8791, 0.7313, 0.8894, 0.5243, 0.8170, 0.3591, 0.7307, 0.2540, 0.6320,
      0.1565, 0.5842, 0.4731, 0.3779, 0.4000, 0.2746, 0.3470, 0.1822, 0.2813,
      0.5301, 0.6058, 0.3245, 0.5183, 0.2218, 0.4467, 0.1390, 0.3602, 0.5128,
      0.7209, 0.3284, 0.6492, 0.2378, 0.5819, 0.1616, 0.5043, 0.5231, 0.8182,
      0.3817, 0.7824, 0.3063, 0.7379, 0.2324, 0.6848
    };  // 21 * 2
    return skeleton;
}

/**
 * @brief 
 * 
 * @param gestureID
 * @return std::string 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
std::string gestureMapping(const int gestureID) {
  switch (gestureID) {
      case 0:
          return std::string("num 0: Neg");
           break;
      case 1:
          return std::string("num 1: Throw Up");
          break;
      case 2:
          return std::string("num 2: Throw Down");
          break;
      case 3:
          return std::string("num 3: Throw Left");
          break;
      case 4:
          return std::string("num 4: Throw Right");
          break;
      case 5:
          return std::string("num 5: Zoom In");
          break;
      case 6:
          return std::string("num 6: Zoom Out");
          break;
      case 999:
          return std::string("please wait cold start");
          break;
      default:
          return std::string("unknown");
          break;
  }
}

/**
 * @brief 
 * 
 * @param inputFrame 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
cv::Mat drawGestureResult(const cv::Mat &inputFrame, InferenceResult *result) {
  // current gesture
  std::string gestureResultText = gestureMapping(result->gestureID);
  std::string showText = " Gesture Result: " + gestureResultText;
  cv::Mat outputFrame = inputFrame.clone();
  cv::putText(
    outputFrame, showText, cv::Point(outputFrame.rows / 10, outputFrame.cols / 10), 
    cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2
  );  // Blue, Green, Red

  // current boundingbox
  if (result->isHandDetected) {
    cv::rectangle(
      outputFrame, 
      cv::Point(result->handBoundingBoxOnFrame[0], result->handBoundingBoxOnFrame[1]), 
      cv::Point(result->handBoundingBoxOnFrame[2], result->handBoundingBoxOnFrame[3]), 
      cv::Scalar(0, 255, 0), 2
    );  // Blue, Green, Red
    std::string strProb = std::to_string(result->probability);
    cv::putText(
      outputFrame, strProb, cv::Point(result->handBoundingBoxOnFrame[0], result->handBoundingBoxOnFrame[1]), 
      cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2
    );  // Blue, Green, Red
    for(int i = 0; i<result->currentSkeletonOnFrame->size(); i = i + 2){
      cv::circle(
        outputFrame, cv::Point(result->currentSkeletonOnFrame->at(i), result->currentSkeletonOnFrame->at(i + 1)), 
        3, cv::Scalar(0, 255, 0), -1
      );  // 画点，其实就是实心圆
	  }
  }
  return outputFrame;
}

int main(int argc, char **argv) {
  samplesCommon::Args args;
  bool argsOK = samplesCommon::parseArgs(args, argc, argv);
  if (!argsOK) {
    sample::gLogError << " Invalid arguments " << std::endl;
    printHelpInfo();
    return EXIT_FAILURE;
  }
  if (args.help) {
    printHelpInfo();
    return EXIT_SUCCESS;
  }
  auto sampleTest = sample::gLogger.defineTest(SampleName, argc, argv);
  sample::gLogger.reportTestStart(sampleTest);
  // make unique ptr to call class member functions to execute inference
  std::unique_ptr<YOLO> ptrYOLO = std::make_unique<YOLO>(YOLO::initializeSampleParams(args));
  std::unique_ptr<Skeleton> ptrSkeleton = std::make_unique<Skeleton>(Skeleton::initializeSampleParams(args));
  std::unique_ptr<Gesture> ptrGesture = std::make_unique<Gesture>(Gesture::initializeSampleParams(args));
  sample::gLogInfo << " Building and running a GPU inference engine for Dynamic Gesture Detection " << std::endl;
  if (!ptrYOLO->build()) {return sample::gLogger.reportFail(sampleTest);}
  if (!ptrSkeleton->build()) {return sample::gLogger.reportFail(sampleTest);}
  if (!ptrGesture->build()) {return sample::gLogger.reportFail(sampleTest);}
  cv::Mat frame;
  cv::Mat croppedFrame;
  InferenceResult result;
#if METHOD != 2  // camera start
  cv::VideoWriter writer;
  #if METHOD == 0
    std::cout << "------------- Read from camera!-------------" << std::endl;
    cv::VideoCapture capture(0);
    writer.open(
      "../data/camera_test.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, 
      cv::Size(960, 960), true
    );
  #elif METHOD == 1
    std::cout << "------------- Read from video!-------------" << std::endl;
    cv::VideoCapture capture("../data/test.mp4");
    double frameRate = capture.get(cv::CAP_PROP_FPS);
    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    writer.open(
      "../data/video_test.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), frameRate, 
      cv::Size(height, height), true
    );
  #endif
  if (!capture.isOpened()) {
    std::cout << "video not open." << std::endl;
    return 1;
  }
  bool stop(false);
  int frameCounter = 0;
  std::string fileDir = "../data/video_result_frames/";
  while (!stop) {
    frameCounter++;
    if (!capture.read(frame)) {
      std::cout << "no video frame" << std::endl;
      break;
    }
    if (frame.data == nullptr) {
      std::cout << "[Warning] Reading image failed!" << std::endl;
      return 0;
    }
    std::cout << " Time for inference started " << std::endl;
    // record start time of inference
    auto infer_start = std::chrono::system_clock::now();
    cv::Rect ROI((frame.cols - frame.rows) / 2, 0, frame.rows, frame.rows);  // x, y, w, h, for width (cols) > height (rows)
    // cv::Rect ROI(0, (frame.rows - frame.cols) / 2, frame.cols, frame.cols);  // x, y, w, h, for width (cols) < height (rows)
    croppedFrame = frame(ROI);
    // test image
    cv::Mat grayFrame;
    cv::cvtColor(croppedFrame, grayFrame, cv::COLOR_BGR2GRAY);
    std::cout << "hand detection" << std::endl;
    if (!ptrYOLO->infer(grayFrame, &result)) {return sample::gLogger.reportFail(sampleTest);}
    std::cout << "skeleton detection" << std::endl;
    if (!ptrSkeleton->infer(grayFrame, &result)) {return sample::gLogger.reportFail(sampleTest);}
    std::cout << "gesture classification" << std::endl;
    if (!ptrGesture->infer(&result)) {return sample::gLogger.reportFail(sampleTest);}
    // record end time of inference
    auto infer_end = std::chrono::system_clock::now();
    // std::chrono::seconds also works
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start);
    std::cout << " Time for inference " << duration.count()  << " ms"  << std::endl;
    // draw gesture inference result
    cv::Mat outputFrame = drawGestureResult(croppedFrame, &result);
  #if REMOTE == 0
	  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	  cv::imshow("Display window", outputFrame);
    if (cv::waitKey(10) == 'q') {break;}
  #endif
    writer.write(outputFrame);
    std::string filename = fileDir + std::to_string(frameCounter) + ".jpg";
    cv::imwrite(filename, outputFrame);
  }
  std::cout << " final frame number is: " << frameCounter << std::endl;
  capture.release();
  writer.release();
#elif METHOD == 2  // image start
  std::cout << "------------- Read from image!-------------" << std::endl;
  frame = cv::imread("../data/1_ori.jpg");
  // infer 1000 times
  for (int i = 0; i < 10000; i++) {
    std::cout << " Time for inference started " << std::endl;
    // record start time of inference
    auto infer_start = std::chrono::system_clock::now();
    // cv::Rect ROI((frame.cols - frame.rows) / 2, 0, frame.rows, frame.rows);  // x, y, w, h, for width (cols) > height (rows)
    // // cv::Rect ROI(0, (frame.rows - frame.cols) / 2, frame.cols, frame.cols);  // x, y, w, h, for width (cols) < height (rows)
    // croppedFrame = frame(ROI);
    // test image
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    std::cout << "hand detection" << std::endl;
    if (!ptrYOLO->infer(grayFrame, &result)) {return sample::gLogger.reportFail(sampleTest);}
    std::cout << "skeleton detection" << std::endl;
    if (!ptrSkeleton->infer(grayFrame, &result)) {return sample::gLogger.reportFail(sampleTest);}
    std::cout << "gesture classification" << std::endl;
    if (!ptrGesture->infer(&result)) {return sample::gLogger.reportFail(sampleTest);}
    // record end time of inference
    auto infer_end = std::chrono::system_clock::now();
    // std::chrono::seconds also works
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start);
    std::cout << " Time for inference " << duration.count()  << " ms"  << std::endl;
    // draw gesture inference result
    cv::Mat outputFrame = drawGestureResult(frame, &result);
  #if REMOTE == 0
	  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	  cv::imshow("Display window", outputFrame);
	  cv::waitKey(10);
  #endif
    cv::imwrite("../data/save.jpg", outputFrame);
  }
// image end
#endif
  return sample::gLogger.reportPass(sampleTest);
}