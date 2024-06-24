#ifndef INFERENCERESULT_H
#define INFERENCERESULT_H

#include <vector>

struct InferenceResult
{
    int gestureID;
    float probability;
    std::vector<float> *currentSkeletonOnFrame;
    std::vector<float> *currentScaledSkeletonOnFrame;
    int *handBoundingBoxOnFrame;
    int *handBoundingBoxOnYOLOInput;
    bool isHandDetected;
};

#endif