#ifndef GESTURE_HPP
#define GESTURE_HPP

#include "Net.hpp"

/**
 * @brief Gesture class
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
class Gesture : public Net {
public:
  /**
   * @brief Construct a new Gesture object
   * 
   * @param params 
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  Gesture(const samplesCommon::OnnxSampleParams &params) : Net(params) {}
  /**
   * @brief Destroy the Gesture object (virtual destructor)
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  virtual ~Gesture() {}

  static samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args &args);
  bool infer(const cv::Mat &img, InferenceResult *result);
  bool infer(InferenceResult *result);

private:
  bool processOpenCVInput_(const cv::Mat &img, const samplesCommon::BufferManager &buffers);
  bool parseOutput_(const samplesCommon::BufferManager &buffers, InferenceResult *result);

  std::queue<std::vector<float>> gestureQueue_;
  int gestureID_;
  int frameCounter_ = 0;
  const int coldStartFrameNum_ = 16;
};

/**
 * @brief 
 * 
 * @param args 
 * @return samplesCommon::OnnxSampleParams 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
samplesCommon::OnnxSampleParams Gesture::initializeSampleParams(const samplesCommon::Args &args) {
  samplesCommon::OnnxSampleParams params;
  params.dataDirs.push_back("../model");
  params.onnxFileName = "cls_cutid.trt";
  params.inputTensorNames.push_back("input");
  params.outputTensorNames.push_back("out");
  params.dlaCore = args.useDLACore;
  params.int8 = args.runInInt8;
  params.fp16 = args.runInFp16;
  // params.preprocess_device = args.preprocess_device;
  return params;
}

/**
 * @brief 
 * 
 * @param img 
 * @param buffers 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Gesture::processOpenCVInput_(const cv::Mat &img, const samplesCommon::BufferManager &buffers) {return true;}

/**
 * @brief 
 * 
 * @param img 
 * @param result 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Gesture::infer(const cv::Mat &img, InferenceResult *result) {
  std::cout << " infer(const cv::Mat &img, InferenceResult *result) " << std::endl;
  return false;
}

/**
 * @brief 
 * 
 * @param result 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Gesture::infer(InferenceResult *result) {
  if (result == nullptr) {return false;}
  // check if yolo detected hand normally or not
  // if (!result->isHandDetected) {return true;}  // temporarily comment this line
  // for cold start
  if (frameCounter_ < coldStartFrameNum_) {
    result->gestureID = 999;
    gestureQueue_.push(*(result->currentScaledSkeletonOnFrame));
    frameCounter_++;
    return true;
  }
  frameCounter_++;
  samplesCommon::BufferManager buffers(mEngine_);
  std::cout << " start to call .inferFeat() " << std::endl;
  auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine_->createExecutionContext());
  if (!context) {return false;}
  // Read the input data into the managed buffers
  assert(mParams_.inputTensorNames.size() == 1);
  // record start time of cuda
  auto cuda_time_start = std::chrono::system_clock::now();
  bool bret = true;
  const int kDevice = 0;
  if (kDevice == 0) {
    bret = processQueueInput_(*(result->currentScaledSkeletonOnFrame), buffers, gestureQueue_);
    buffers.copyInputToDevice();
  }
  if (!bret) {return false;}
  // Memcpy from host input buffers to device input buffers
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status) {return false;}
  // Memcpy from device output buffers to host output buffers
  buffers.copyOutputToHost();
  // record start time of cuda
  auto cuda_time_end = std::chrono::system_clock::now();
  // std::chrono::seconds also works
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_time_end - cuda_time_start);
  std::cout << " ++++++++ " << " cuda time period " << duration.count() << " ms " << std::endl;
  // Verify results
  if (!parseOutput_(buffers, result)) {return false;}
  std::cout << "end gesture inference!" << std::endl;
  return true;
}

/**
 * @brief 
 * 
 * @param buffers 
 * @param result 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Gesture::parseOutput_(const samplesCommon::BufferManager &buffers, InferenceResult *result) {
  if (result == nullptr) {return false;}
  const int outputSize = mOutputDims_.d[1];
  float *output = static_cast<float *>(buffers.getHostBuffer(mParams_.outputTensorNames[0]));
  const int kDevice = 0;
  // get gesturID
  // Calculate Softmax sum
  float sum{0.0f};
  for (int i = 0; i < outputSize; i++) {
    output[i] = exp(output[i]);
    sum += output[i];
  }
  int max_idx = -1;
  float max_prob = 0.0;
  sample::gLogInfo << "Output:" << std::endl;
  for (int i = 0; i < outputSize; i++) {
    output[i] /= sum;
    std::cout << i << "-->" << output[i] << std::endl;
    if (max_prob < output[i]) {
      max_prob = output[i];
      max_idx = i;
    }
  }
  result->gestureID = max_idx;
  sample::gLogInfo << " Prob " << std::fixed << std::setw(5) << std::setprecision(4) << output[max_idx] 
                   << " " << " Class " << max_idx << " "
                   << std::string(int(std::floor(output[max_idx] * 10 + 0.5f)), '*') << std::endl;
  // sample::gLogInfo << std::endl;
  if (max_prob > 0.8f) {std::cout<< " [Note] gesture inference result is of a low confidence " << std::endl;}
  return true;
}

#endif