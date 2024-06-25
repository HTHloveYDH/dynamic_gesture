#ifndef NET_HPP
#define NET_HPP

#include <opencv2/opencv.hpp>

#include <iostream>
#include <queue>
#include <vector>
#include <chrono>

#include "NvInfer.h"
#include "buffers.h"
#include "argsParser.h"
#include "sampleEngines.h"
#include "common.h"
#include "logger.h"

#include "InferenceResult.h"

/**
 * @brief Net class
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
class Net {
template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
  /**
   * @brief Construct a new Net instance
   * @param params 
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  Net(const samplesCommon::OnnxSampleParams &params) : mParams_(params), mEngine_(nullptr) {}
  /**
   * @brief Destroy the Net object (virtual destructor)
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  virtual ~Net() {}

  bool build();
  virtual bool infer(const cv::Mat &img, InferenceResult *result) = 0;
  virtual bool infer(InferenceResult *result) = 0;

protected:
  // TODO: this function shoule also be virtual and be implemented in 'Class Gesture'
  bool processQueueInput_(
    std::vector<float> &feat, const samplesCommon::BufferManager &buffers, std::queue<std::vector<float>> &inputsQueue
  );
  virtual bool processOpenCVInput_(const cv::Mat &img, const samplesCommon::BufferManager &buffers) = 0;
  virtual bool parseOutput_(const samplesCommon::BufferManager &buffers, InferenceResult *result) = 0;
  
  samplesCommon::OnnxSampleParams mParams_;  //!< The parameters for the sample.
  nvinfer1::Dims mInputDims_;   //!< The dimensions of the input to the network.
  nvinfer1::Dims mOutputDims_;  //!< The dimensions of the output to the network.
  int mNumber_ = 0;             //!< The number to classify
  int inputC_ = 0;
  int inputH_ = 0;
  int inputW_ = 0;
  int inputB_ = 0;
#if METHOD == 0  // camera
  const float initImgWidth_ = 960.0;
  const float initImgHeight_ = 960.0;
#elif METHOD == 1  // video
  const float initImgWidth_ = 720.0;
  const float initImgHeight_ = 720.0;
#elif METHOD == 2  // single image
  const float initImgWidth_ = 256.0;
  const float initImgHeight_ = 256.0;
#endif
  // TODO: this member data should be defined in 'Class Gesture'
  std::vector<float> nInputBufferVector_;
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine_;  //!< The TensorRT engine used to run the network
};

/**
 * @brief 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Net::build() {
#ifdef PLATFORM
  #if PLATFORM == 1  // on Jetson
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");  // only for Jetson
  #endif
#endif
  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder) {return false;}
  std::string onnx_file = locateFile(mParams_.onnxFileName, mParams_.dataDirs);
  mEngine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      sample::loadEngine(onnx_file, mParams_.dlaCore, sample::gLogError),
      samplesCommon::InferDeleter());
  if (!mEngine_) {return false;}
  int index = mEngine_->getBindingIndex(mParams_.inputTensorNames.at(0).c_str());
  if (index == -1) {return false;}
  mInputDims_ = mEngine_->getBindingDimensions(index);
  assert(mInputDims_.nbDims == 4);
  index = mEngine_->getBindingIndex(mParams_.outputTensorNames.at(0).c_str());
  if (index == -1) {return false;}
  mOutputDims_ = mEngine_->getBindingDimensions(index);
  // assert(mOutputDims_.nbDims == 3);
  inputB_ = mInputDims_.d[0];
  inputC_ = mInputDims_.d[1];
  inputH_ = mInputDims_.d[2];
  inputW_ = mInputDims_.d[3];
  // print inout dims info
  std::cout << " input dim info: " << std::endl;
  std::cout << " inputB_: " << inputB_ << std::endl;
  std::cout << " inputH_: " << inputH_ << std::endl;
  std::cout << " inputW_: " << inputW_ << std::endl;
  std::cout << " inputC_: " << inputC_ << std::endl;
  return true;
}

/**
 * @brief 
 * 
 * @param feat 
 * @param buffers 
 * @param inputsQueue 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Net::processQueueInput_(std::vector<float> &feat, const samplesCommon::BufferManager &buffers, \
                             std::queue<std::vector<float>> &inputsQueue) {
  float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams_.inputTensorNames[0]));
  // 出队 & 入队
  inputsQueue.pop();
  inputsQueue.push(feat);
  // clear nInputBufferVector_ (nInputBufferVector_.size() == 0)
  nInputBufferVector_.clear();
  // tranverse queue by repeat push() & pop()
  for(int i = 0; i < inputC_; i++) {   // size 必须是固定值
    for (const auto it : inputsQueue.front()) {
      nInputBufferVector_.push_back(it);
    }
    inputsQueue.push(inputsQueue.front());
    inputsQueue.pop();
  }
  int volData = inputC_ * inputH_ * inputW_;
  memcpy(hostDataBuffer, nInputBufferVector_.data(), sizeof(float) * inputB_ * volData);
  return true;
}

#endif
