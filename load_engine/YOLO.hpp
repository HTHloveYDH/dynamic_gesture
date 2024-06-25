#ifndef YOLO_HPP
#define YOLO_HPP

#include "Net.hpp"
#include "hungarianTrack.h"  // currently #include <Eigen/Dense> needed

/**
 * @brief YOLO class
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
class YOLO : public Net {
public:
  /**
   * @brief Construct a new YOLO object
   * 
   * @param params 
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  YOLO(const samplesCommon::OnnxSampleParams &params);
  /**
   * @brief Destroy the YOLO object (virtual destructor)
   * @author Tinghao.Han (Tinghao.Han@harman.com)
   */
  virtual ~YOLO() {}

  static samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args &args);
  bool infer(const cv::Mat &img, InferenceResult *result);
  bool infer(InferenceResult *result);

private:
  bool processOpenCVInput_(const cv::Mat &img, const samplesCommon::BufferManager &buffers);
  bool parseOutput_(const samplesCommon::BufferManager &buffers, InferenceResult *result);
  void postProcessCoordinate_(std::vector<float> &box);
  void filterPredictions_(float *output, std::vector<std::vector<float>> &boxVector, int bufferSize);
  static float iou_(std::vector<float> box1, std::vector<float> box2);
  static bool sortBBox_(std::vector<float> box1, std::vector<float> box2);
  std::vector<std::vector<float>> nms_(std::vector<std::vector<float>> &boxVector);
  float sigmoid_(float x);
  void createGridCellOnFeatureMap_();
  void createAnchorOnFeatureMap_();
  void getYOLOPredictions_(
    float *output, int numGridCell, int stride, std::vector<std::pair<int, int>> &gridCellVector, \
    float *featAnchorSize
  );

  int handBoundingBoxOnFrame_[4];
  int handBoundingBoxOnYOLOInput_[4];
  const float nmsIOUThresh_ = 0.6;
  const float confThresh_ = 0.2;
  const float minHWThresh_ = 4;
  const float inputSize_ = 128.0;
  const float xRatio_ = inputSize_ / initImgWidth_;
  const float yRatio_ = inputSize_ / initImgHeight_;
  const float boxDim_ = 6;
  const int numAnchor_ = 3;
  const int numGridCell1_ = 16;
  const int numGridCell2_ = 32;
  const int stride1_ = inputSize_ / float(numGridCell1_);  // feature map size: 16 * 16
  const int stride2_ = inputSize_ / float(numGridCell2_);  // feature map size: 32 * 32
  const float imgAnchorSize1_[6] = {24.0, 25.0, 41.0, 52.0, 105.0, 98.0};  // feature map size: 16 * 16
  const float imgAnchorSize2_[6] = {3.0, 4.0, 7.0, 8.0, 11.0, 17.0};  // feature map size: 32 * 32
  float featAnchorSize1_[6];  // feature map size: 16 * 16
  float featAnchorSize2_[6];  // feature map size: 32 * 32
  std::vector<std::pair<int, int>> gridCellVector1_;
  std::vector<std::pair<int, int>> gridCellVector2_;
};

/**
 * @brief Construct a new YOLO object
 * 
 * @param params 
 */
YOLO::YOLO(const samplesCommon::OnnxSampleParams &params) : Net(params) {
  createGridCellOnFeatureMap_(); 
  createAnchorOnFeatureMap_();
}

/**
 * @brief 
 * 
 * @param args 
 * @return samplesCommon::OnnxSampleParams 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
samplesCommon::OnnxSampleParams YOLO::initializeSampleParams(const samplesCommon::Args &args) {
  samplesCommon::OnnxSampleParams params;
  params.dataDirs.push_back("../model");
  params.onnxFileName = "hand_detection_rm_scatter.trt";
  params.inputTensorNames.push_back("input");
  params.outputTensorNames.push_back("out1");  // feature map size 16 * 16
  params.outputTensorNames.push_back("out2");  // feature map size 32 * 32
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
bool YOLO::processOpenCVInput_(const cv::Mat &img, const samplesCommon::BufferManager &buffers) {
  float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams_.inputTensorNames[0]));
  std::cout << "--->>> Processing of opencv!" << std::endl;
  // resize to inputSize * inputSize
  cv::Mat resizedImg;
  cv::resize(img, resizedImg, cv::Size(inputSize_, inputSize_), (0.0), (0.0), 0);  // interpolation: INTER_LINEAR
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
bool YOLO::infer(const cv::Mat &img, InferenceResult *result) {
  if (result == nullptr) {return false;}
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
    bret = processOpenCVInput_(img, buffers);
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
  std::cout << "end yolo inference!" << std::endl;
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
bool YOLO::infer(InferenceResult *result) {
  std::cout << " infer(InferenceResult *result) is not supported in YOLO " << std::endl;
  return false;
}

void YOLO::filterPredictions_(float *output, std::vector<std::vector<float>> &boxVector, int bufferSize) {
  float leftTopX, leftTopY, rightBottomX, rightBottomY, W, H, conf;
  for(int i = 0; i < bufferSize; i += 6) {
    output[i + 2] = output[i + 3] = std::max(output[i + 2], output[i + 3]);
    W = output[i + 2];
    H = output[i + 3];
    leftTopX = output[i] - output[i + 2] / 2;
    leftTopY = output[i + 1] - output[i + 3] / 2;
    rightBottomX = output[i] + output[i + 2] / 2;
    rightBottomY = output[i + 1] + output[i + 3] / 2;
    conf = output[i + 4];
    if (conf < confThresh_) {continue;}
    if (W < minHWThresh_) {continue;}
    if (H < minHWThresh_) {continue;}
    // the following four jugments will restrict to the hands that are around the center of current frame
    // if (leftTopX < 0) {continue;}
    // if (leftTopY < 0) {continue;}
    // if (rightBottomX > inputSize_ - 1.0) {continue;}
    // if (rightBottomY > inputSize_ - 1.0) {continue;}
    // the following four jugments is friendly to the hands that are close to the edge of current frame
    if (leftTopX < 0) {leftTopX = 0.0;}
    if (leftTopY < 0) {leftTopY = 0.0;}
    if (rightBottomX > inputSize_ - 1.0) {rightBottomX = inputSize_ - 1.0;}
    if (rightBottomY > inputSize_ - 1.0) {rightBottomY = inputSize_ - 1.0;}
    // xywh score class --> left_top(x1,y1) right_bottom(x2,y2) score
    std::vector<float> box;
    box.push_back(leftTopX);  // left_top x
    box.push_back(leftTopY);  // left_top y
    box.push_back(rightBottomX);  // right_bottom x
    box.push_back(rightBottomY);  // right_bottom y
    box.push_back(conf);  // score 
    boxVector.push_back(box);
  }
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
bool YOLO::parseOutput_(const samplesCommon::BufferManager &buffers, InferenceResult *result) {
  if (result == nullptr) {return false;}
  // const int outputSize = mOutputDims_.d[1];
  const int kDevice = 0;
  std::vector<std::vector<float>> boxVector;
  float *output1 = static_cast<float *>(buffers.getHostBuffer(mParams_.outputTensorNames[0]));
  getYOLOPredictions_(output1, numGridCell1_, stride1_, gridCellVector1_, featAnchorSize1_);
  filterPredictions_(output1, boxVector, numAnchor_ * numGridCell1_ * numGridCell1_ * boxDim_);  // feature map size 16 * 16
  float *output2 = static_cast<float *>(buffers.getHostBuffer(mParams_.outputTensorNames[1]));
  getYOLOPredictions_(output2, numGridCell2_, stride2_, gridCellVector2_, featAnchorSize2_);
  filterPredictions_(output2, boxVector, numAnchor_ * numGridCell2_ * numGridCell2_ * boxDim_);  // feature map size 32 * 32
  std::cout << "bbx num befor nms bbx num befor nms bbx num befor nms bbx num befor nms: " << boxVector.size() << std::endl;
  std::vector<std::vector<float>> afterNMSBoxVector = nms_(boxVector);
  std::cout << "bbx num after nms bbx num after nms bbx num after nms bbx num after nms: " << afterNMSBoxVector.size() << std::endl;
  if (afterNMSBoxVector.size() == 0){
    result->handBoundingBoxOnFrame = handBoundingBoxOnFrame_;  // 如果YOLO没有检测到手
    result->handBoundingBoxOnYOLOInput = handBoundingBoxOnYOLOInput_;
    result->isHandDetected = false;
    result->probability = 0.0;
    std::cout << " No hand detected, skip this frame !!! " << std::endl;
  }
  else if (afterNMSBoxVector.size() == 1){
    postProcessCoordinate_(afterNMSBoxVector[0]);;  // 如果YOLO检测到一只手
    result->handBoundingBoxOnFrame = handBoundingBoxOnFrame_;
    result->handBoundingBoxOnYOLOInput = handBoundingBoxOnYOLOInput_;
    result->isHandDetected = true;
    result->probability = afterNMSBoxVector[0][4];
  }
  else{
    // TODO: multi-hand suppression
#if HAND_TRACKING == 1
    std::vector<float> box{
      float(result->handBoundingBoxOnYOLOInput[0]), float(result->handBoundingBoxOnYOLOInput[1]), 
      float(result->handBoundingBoxOnYOLOInput[2]), float(result->handBoundingBoxOnYOLOInput[3]), 
      float(result->probability), 0.0
    };
    std::vector<std::pair<size_t, size_t>> assignments = hungarianTrack(box, afterNMSBoxVector, iou_);
#endif
    // tmp solution: track hand via hungarian algorthm
    postProcessCoordinate_(afterNMSBoxVector[assignments[0].second]);  // 如果YOLO检测到多只手
    result->handBoundingBoxOnFrame = handBoundingBoxOnFrame_;
    result->handBoundingBoxOnYOLOInput = handBoundingBoxOnYOLOInput_;
    result->isHandDetected = true;
    result->probability = afterNMSBoxVector[assignments[0].second][4];
  }
  return true;
}

/**
 * @brief 
 * 
 * @param box 
 * Tinghao.Han (Tinghao.Han@harman.com)
 */
void YOLO::postProcessCoordinate_(std::vector<float> &box) {
  int finalBoxOnFrame[4] = {
    int(box[0] / xRatio_), int(box[1] / yRatio_), int(box[2] / xRatio_), int(box[3] / yRatio_)
  };
  int finalBoxOnYOLOInput[4] = {int(box[0]), int(box[1]), int(box[2]), int(box[3])};
  memcpy(handBoundingBoxOnFrame_, finalBoxOnFrame, sizeof(finalBoxOnFrame));
  memcpy(handBoundingBoxOnYOLOInput_, finalBoxOnYOLOInput, sizeof(finalBoxOnYOLOInput));
  std::cout << "hand left-top point X detected be yolo: " << box[0] 
            << " , " << finalBoxOnFrame[0] << std::endl;
  std::cout << "hand left-top point Y detected be yolo: " << box[1] 
            << " , " << finalBoxOnFrame[1] << std::endl;
  std::cout << "hand right-bottom-top X point detected be yolo: " << box[2] 
            << " , " << finalBoxOnFrame[2] << std::endl;
  std::cout << "hand right-bottom point Y detected be yolo: " << box[3] 
            << " , " << finalBoxOnFrame[3] << std::endl;
  std::cout << "confidence: " << box[4] << std::endl;
}

/**
 * @brief 
 * 
 * @param box1 
 * @param box2 
 * @return float 
 * @author Mengmeng.Xiao (Mengmeng.Xiao@harman.com)
 */
float YOLO::iou_(std::vector<float> box1, std::vector<float> box2) {
  float max_x = std::max(box1[0], box2[0]);  
  float min_x = std::min(box1[2], box2[2]);  
  float max_y = std::max(box1[1], box2[1]);
  float min_y = std::min(box1[3], box2[3]);
  if(min_x <= max_x || min_y <= max_y) {return 0;}  // 没有重叠
  float over_area = (min_x - max_x) * (min_y - max_y);  
  float area_a = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  float area_b = (box2[2] - box2[0]) * (box2[3] - box1[1]);
  float iou = over_area / (area_a + area_b - over_area);
  return iou;
}

/**
 * @brief 
 * 
 * @param box1 
 * @param box2 
 * @return true 
 * @return false 
 * @author Mengmeng.Xiao (Mengmeng.Xiao@harman.com)
 */
bool YOLO::sortBBox_(std::vector<float> box1, std::vector<float> box2) {
  return (box1[4] > box2[4]);
}

/**
 * @brief 
 * 
 * @param boxVector 
 * @return vector<vector<float>> 
 * @author Mengmeng.Xiao (Mengmeng.Xiao@harman.com)
 */
std::vector<std::vector<float>> YOLO::nms_(std::vector<std::vector<float>> &boxVector) {
  std::vector<std::vector<float>>  res;
  int erase_cnt = 0;
  while(boxVector.size() - erase_cnt > 0) {
    sort(boxVector.begin(), boxVector.end(), sortBBox_);
    res.push_back(boxVector[0]);
    for(int i = 0; i < boxVector.size() - erase_cnt - 1; i++) {
      float iou_value = iou_(boxVector[0], boxVector[i+1]);
      if (iou_value > nmsIOUThresh_) {
        boxVector[i+1][4] = 0.0;  // 将待删除的box置信度赋值0
        erase_cnt++;
      }
    }
    boxVector[0][4] = 0.0;  // 将待删除的box置信度赋值0
    erase_cnt++;
  }
  return res;
}

/**
 * @brief 
 * 
 * @param x 
 * @return float 
 * @author Mengmeng.Xiao (Mengmeng.Xiao@harman.com)
 */
float YOLO::sigmoid_(float x) {
  return (1 / (1 + exp(-x)));
}

/**
 * @brief Create a grid cells
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
void YOLO::createGridCellOnFeatureMap_() {
  for (int row = 0; row < numGridCell1_; row++){
    for (int col = 0; col < numGridCell1_; col++){
      std::pair<int, int> GridCell(col, row);
      gridCellVector1_.push_back(GridCell);
    }
  }
  for (int row = 0; row < numGridCell2_; row++){
    for (int col = 0; col < numGridCell2_; col++){
      std::pair<int, int> GridCell(col, row);
      gridCellVector2_.push_back(GridCell);
    }
  }
}

/**
 * @brief Get the Feat Anchor Size object
 * @author Mengmeng.Xiao (Mengmeng.Xiao@harman.com)
 */
void YOLO::createAnchorOnFeatureMap_() {
  for (int i = 0; i < 6; i++) {
    featAnchorSize1_[i] = imgAnchorSize1_[i] / stride1_;
    featAnchorSize2_[i] = imgAnchorSize2_[i] / stride2_;
  }
}

/**
 * @brief 
 * @author Tinghao.Han (Tinghao.Han@harman.com)
 */
void YOLO::getYOLOPredictions_(float *output, int numGridCell, int stride, std::vector<std::pair<int, int>> &gridCellVector, \
                               float *featAnchorSize) {
  for (int i = 0; i < numAnchor_ * numGridCell * numGridCell; i++) {
    int anchorIndexOnFeatureMap = i / numGridCell / numGridCell;
    int rowIndexOnFeatureMap = (i / numGridCell) % numGridCell; 
    int colIndexOnFeatureMap = i % numGridCell; 
    output[i * 6] = sigmoid_(output[i * 6]) + float(gridCellVector[rowIndexOnFeatureMap * numGridCell + colIndexOnFeatureMap].first); 
    output[i * 6 + 1] = sigmoid_(output[i * 6 + 1]) + float(gridCellVector[rowIndexOnFeatureMap * numGridCell + colIndexOnFeatureMap].second);  
    output[i * 6 + 2] = exp(output[i * 6 + 2]) * featAnchorSize[anchorIndexOnFeatureMap * (numAnchor_ - 1)]; 
    output[i * 6 + 3] = exp(output[i * 6 + 3]) * featAnchorSize[anchorIndexOnFeatureMap * (numAnchor_ - 1) + 1];  
    output[i * 6] = stride * output[i * 6];
    output[i * 6 + 1] = stride * output[i * 6 + 1];
    output[i * 6 + 2] = stride * output[i * 6 + 2];
    output[i * 6 + 3] = stride * output[i * 6 + 3];
    output[i * 6 + 4] = sigmoid_(output[i * 6 + 4]);  
  }
}

#endif
