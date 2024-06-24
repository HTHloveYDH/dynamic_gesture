#include <fstream>
#include <iostream>
// #include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

std::vector<std::string> load_labels(std::string labels_file)
{
    std::ifstream file(labels_file.c_str());
    if (!file.is_open())
    {
        fprintf(stderr, "unable to open label file\n");
        exit(-1);
    }
    std::string label_str;
    std::vector<std::string> labels;

    while (std::getline(file, label_str))
    {
        if (label_str.size() > 0)
            labels.push_back(label_str);
    }
    file.close();
    return labels;
}

int main(int argc, char **argv)
{

    // Get Model label and input image
    if (argc != 4)
    {
        fprintf(stderr, "TfliteClassification.exe modelfile labels image\n");
        exit(-1);
    }
    const char *modelFileName = argv[1];
    const char *labelFile = argv[2];
    const char *imageFile = argv[3];

    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr)
    {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    // Initiate Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter\n");
        exit(-1);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }
    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);
    // Get Input Tensor Dimensions
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];
    // Load Input Image
    cv::Mat image;
    auto frame = cv::imread(imageFile);
    if (frame.empty())
    {
        fprintf(stderr, "Failed to load iamge\n");
        exit(-1);
    }

    // Copy image to input tensor
    cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
    memcpy(interpreter->typed_input_tensor<unsigned char>(0), image.data, image.total() * image.elemSize());

    // Inference
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    interpreter->Invoke();
    end = std::chrono::steady_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Get Output
    int output = interpreter->outputs()[0];
    TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    std::vector<std::pair<float, int>> top_results;
    float threshold = 0.01f;

    switch (interpreter->tensor(output)->type)
    {
    case kTfLiteInt32:
        tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size, 1, threshold, &top_results, kTfLiteFloat32);
        break;
    case kTfLiteUInt8:
        tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &top_results, kTfLiteUInt8);
        break;
    default:
        fprintf(stderr, "cannot handle output type\n");
        exit(-1);
    }
    // Print inference ms in input image
    cv::putText(frame, "Infernce Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    // Load Labels
    auto labels = load_labels(labelFile);

    // Print labels with confidence in input image
    for (const auto &result : top_results)
    {
        const float confidence = result.first;
        const int index = result.second;
        std::string output_txt = "Label :" + labels[index] + " Confidence : " + std::to_string(confidence);
        cv::putText(frame, output_txt, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }

    // Display image
    // cv::imshow("Output", frame);
    // cv::waitKey(0);
    cv::imwrite("../data/save.jpg", frame);
    std::cout << " tflite model inferenced successfully " << std::endl;

    return 0;
}


// #include <cstdio>
// #include <iostream>

// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"

// // This is an example that is minimal to read a model
// // from disk and perform inference. There is no data being loaded
// // that is up to you to add as a user.
// //
// // NOTE: Do not add any dependencies to this that cannot be built with
// // the minimal makefile. This example must remain trivial to build with
// // the minimal build tool.
// //
// // Usage: minimal <tflite model>

// #define TFLITE_MINIMAL_CHECK(x)                              \
//   if (!(x)) {                                                \
//     fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
//     exit(1);                                                 \
//   }

// int main(int argc, char* argv[]) {
//   if (argc != 2) {
//     fprintf(stderr, "minimal <tflite model>\n");
//     return 1;
//   }
//   const char* filename = argv[1];

//   // Load model
//   std::unique_ptr<tflite::FlatBufferModel> model =
//       tflite::FlatBufferModel::BuildFromFile(filename);
//   TFLITE_MINIMAL_CHECK(model != nullptr);

//   // Build the interpreter with the InterpreterBuilder.
//   // Note: all Interpreters should be built with the InterpreterBuilder,
//   // which allocates memory for the Interpreter and does various set up
//   // tasks so that the Interpreter can read the provided model.
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   tflite::InterpreterBuilder builder(*model, resolver);
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   builder(&interpreter);
//   TFLITE_MINIMAL_CHECK(interpreter != nullptr);

//   // Allocate tensor buffers.
//   TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
//   printf("=== Pre-invoke Interpreter State ===\n");
//   tflite::PrintInterpreterState(interpreter.get());

//   // Fill input buffers
//   // TODO(user): Insert code to fill input tensors.
//   // Note: The buffer of the input tensor with index `i` of type T can
//   // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

//   // Run inference
//   TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
//   printf("\n\n=== Post-invoke Interpreter State ===\n");
//   tflite::PrintInterpreterState(interpreter.get());

//   // Read output buffers
//   // TODO(user): Insert getting data out code.
//   // Note: The buffer of the output tensor with index `i` of type T can
//   // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
//   std::cout << "Tflite inference succeed" << std::endl;

//   return 0;
// }