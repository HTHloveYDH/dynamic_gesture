#ifndef SKELETON_HPP
#define SKELETON_HPP

#include "Net.hpp"

/**
 * @brief Skeleton class
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
class Skeleton : public Net {
public:
  /**
   * @brief Construct a new Skeleton object
   * 
   * @param params 
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  Skeleton(const samplesCommon::OnnxSampleParams &params) : Net(params) {}
  /**
   * @brief Destroy the Skeleton object (virtual destructor)
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  virtual ~Skeleton() {}

  static samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args &args);
  bool infer(const cv::Mat &img, InferenceResult *result);
  bool infer(InferenceResult *result);

private:
  bool processOpenCVInput_(const cv::Mat &img, const samplesCommon::BufferManager &buffers);
  bool parseOutput_(const samplesCommon::BufferManager &buffers, InferenceResult *result);
  void postProcessCoordinate_(InferenceResult *result);

  std::vector<float> currentSkeletonOnFrame_;
  std::vector<float> currentScaledSkeletonOnFrame_;
  const float inputSize_ = 64.0;
};

/**
 * @brief 
 * 
 * @param args 
 * @return samplesCommon::OnnxSampleParams 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
samplesCommon::OnnxSampleParams Skeleton::initializeSampleParams(const samplesCommon::Args &args) {
  samplesCommon::OnnxSampleParams params;
  params.dataDirs.push_back("../model");
  params.onnxFileName = "landmark_size-64.trt";
  params.inputTensorNames.push_back("input");
  params.outputTensorNames.push_back("output");
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
bool Skeleton::processOpenCVInput_(const cv::Mat &img, const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams_.inputTensorNames[0]));
  std::cout << "--->>> Processing of opencv!" << std::endl;
  // resize to inputSize * inputSize
  cv::Mat resizedImg;
  cv::resize(img, resizedImg, cv::Size(inputSize_, inputSize_), (0.0), (0.0), 2);  // interpolation: INTER_CUBIC
  // print some information about image size and input size
  std::cout << "inputH_:" << inputH_ << std::endl;
  std::cout << "inputW_:" << inputW_ << std::endl;
  std::cout << "inputC_:" << inputC_ << std::endl;
  std::cout << "resizedImg height:" << resizedImg.rows << std::endl;
  std::cout << "resizedImg width:" << resizedImg.cols << std::endl;
  std::cout << "resizedImg channels::" << resizedImg.channels() << std::endl;
  // normalize image
  cv::Mat normalizedImg = cv::Mat(cv::Size(352, 352), CV_32FC1);
  resizedImg.convertTo(normalizedImg, CV_32FC1, 1 / 255.f);
  std::cout << "normalizedImg is continue?:" << normalizedImg.isContinuous() << std::endl;
  int volChl = inputH_ * inputW_;
  int volImg = inputC_ * volChl;
  memcpy(hostDataBuffer, normalizedImg.data, sizeof(float) * inputB_ * volImg);
  return true;
}

/**
 * @brief 
 * 
 * @param img 
 * @param result 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Skeleton::infer(const cv::Mat &img, InferenceResult *result) {
  if (result == nullptr) {return false;}
  // check if yolo detected hand normally or not
  if (!result->isHandDetected) {
    const int outputSize = mOutputDims_.d[1];  // currently, it is 42
    // clear currentSkeletonOnFrame_ (currentSkeletonOnFrame_.size() == 0)
    currentSkeletonOnFrame_.clear();
    // clear currentScaledSkeletonOnFrame_ (currentScaledSkeletonOnFrame_.size() == 0)
    currentScaledSkeletonOnFrame_.clear();
    // give dummy value (0.0) to currentSkeletonOnFrame_ and currentScaledSkeletonOnFrame_
    for (int i = 0; i < outputSize; i++) {
      currentSkeletonOnFrame_.push_back(0.0);
      currentScaledSkeletonOnFrame_.push_back(0.0);
    }
    result->currentSkeletonOnFrame = &currentSkeletonOnFrame_;
    result->currentScaledSkeletonOnFrame = &currentScaledSkeletonOnFrame_;
    return true;
  }
  samplesCommon::BufferManager buffers(mEngine_);
  std::cout << " start call .inferImg() " << std::endl;
  auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine_->createExecutionContext());
  if (!context) {return false;}
  // Read the input data into the managed buffers
  assert(mParams_.inputTensorNames.size() == 1);
  // record start time of cuda
  auto cuda_time_start = std::chrono::system_clock::now();
  bool bret = true;
  const int kDevice = 0;
  if (kDevice == 0) {
    cv::Rect handBBox(
      result->handBoundingBoxOnFrame[0], result->handBoundingBoxOnFrame[1],  // x, y
      result->handBoundingBoxOnFrame[2] - result->handBoundingBoxOnFrame[0],  // w
      result->handBoundingBoxOnFrame[3] - result->handBoundingBoxOnFrame[1]  // h
    );
    bret = processOpenCVInput_(img(handBBox), buffers);
    buffers.copyInputToDevice();
  }
  if (!bret) {return false;}
  // Memcpy from host input buffers to device input buffers
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status) {return false;}
  // Memcpy from device output buffers to host output buffers
  buffers.copyOutputToHost();
  // record end time of time
  auto cuda_time_end = std::chrono::system_clock::now();
  // std::chrono::seconds also works
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_time_end - cuda_time_start);
  std::cout <<  " ++++++++ " << " cuda time period " << duration.count() << " ms " << std::endl;
  // Verify results
  if (!parseOutput_(buffers, result)) {return false;}
  std::cout << "end skeleton inference!" << std::endl;
  return true;
}

/**
 * @brief 
 * 
 * @param result 
 * @return true 
 * @return false 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
bool Skeleton::infer(InferenceResult *result) {
  std::cout << " infer(InferenceResult *result) is not supported in Skeleton " << std::endl;
  return false;
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
bool Skeleton::parseOutput_(const samplesCommon::BufferManager &buffers, InferenceResult *result) {
  if (result == nullptr) {return false;}
  const int outputSize = mOutputDims_.d[1];  // currently, it is 42
  float *output = static_cast<float *>(buffers.getHostBuffer(mParams_.outputTensorNames[0]));
  const int kDevice = 0;
  // get skeleton
  sample::gLogInfo << "Output:" << std::endl;
  // clear currentSkeletonOnFrame_ (currentSkeletonOnFrame_.size() == 0)
  currentSkeletonOnFrame_.clear();
  // clear currentScaledSkeletonOnFrame_ (currentScaledSkeletonOnFrame_.size() == 0)
  currentScaledSkeletonOnFrame_.clear();
  for (int i = 0; i < outputSize; i++) {
    currentSkeletonOnFrame_.push_back(output[i]);
    currentScaledSkeletonOnFrame_.push_back(output[i]);
  }
  // transfer to image coordinate and normalize
  postProcessCoordinate_(result);
  result->currentSkeletonOnFrame = &currentSkeletonOnFrame_;
  result->currentScaledSkeletonOnFrame = &currentScaledSkeletonOnFrame_;
  return true;
}

/**
 * @brief 
 * 
 * @param input 
 * @param result 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
void Skeleton::postProcessCoordinate_(InferenceResult *result) {
  const int outputSize = mOutputDims_.d[1];  // currently, it is 42
  int xOffset = 0, yOffset = 0;
  float xRatio = 1.0, yRatio = 1.0;
  float boxWidth = result->handBoundingBoxOnFrame[2] - result->handBoundingBoxOnFrame[0];
  float boxHeight = result->handBoundingBoxOnFrame[3] - result->handBoundingBoxOnFrame[1];
  for (int i = 0; i < outputSize; i += 2) {
    xOffset = result->handBoundingBoxOnFrame[0];
    yOffset = result->handBoundingBoxOnFrame[1];
    xRatio = inputSize_ / boxWidth;
    yRatio = inputSize_ / boxHeight;
    currentSkeletonOnFrame_[i] = currentSkeletonOnFrame_[i] * inputSize_ / xRatio + xOffset;
    currentSkeletonOnFrame_[i + 1] = currentSkeletonOnFrame_[i + 1] * inputSize_ / yRatio + yOffset;
    currentScaledSkeletonOnFrame_[i] = currentSkeletonOnFrame_[i] / initImgWidth_;
    currentScaledSkeletonOnFrame_[i + 1] = currentSkeletonOnFrame_[i + 1] / initImgHeight_;
  }
}

#endif
