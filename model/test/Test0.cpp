// g++ -std=c++17 test_fast_scnn.cpp -o test_fast_scnn \
//     -I$CONDA_PREFIX/include \
//     -I$CONDA_PREFIX/include/opencv4 \
//     -L$CONDA_PREFIX/lib -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_dnn \
//     -lpthread -ldl
// LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./test_fast_scnn

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <climits>
#include <cstring>

int main() {

    cv::dnn::Net model = cv::dnn::readNetFromONNX("fast_scnn_pipe.onnx");

        // Load image
        cv::Mat image = cv::imread("input0.png");
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + std::string("input0.png"));
        }

    // prepare model input
    cv::Mat blob = cv::dnn::blobFromImage(image,
        1.f/255.f,
        {256,256});

    // set model input
    model.setInput(blob);

    // get model results
    cv::Mat output = model.forward(); // shape: [1, n_classes, H, W]

    // Get the number of classes, height, and width
    int n_classes = output.size[1];
    int height = output.size[2];
    int width = output.size[3];

    // Get pointer to output data
    const float* data = (float*)output.data;

    // For each pixel, find the class with the highest score (argmax over channels)
    cv::Mat pred_mask(height, width, CV_8U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float max_val = data[y * width + x];
            int max_idx = 0;
            for (int c = 1; c < n_classes; ++c) {
                float val = data[c * height * width + y * width + x];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            pred_mask.at<uchar>(y, x) = static_cast<uchar>(max_idx);
        }
    }

    // Colorize mask for visualization (like matplotlib's "jet" colormap)
    cv::Mat color_mask;
    cv::applyColorMap(pred_mask * (255 / std::max(1, n_classes - 1)), color_mask, cv::COLORMAP_JET);

    // Show original image and mask side by side
    cv::imshow("Original Image", image);
    cv::imshow("Predicted Mask", color_mask);
    cv::waitKey(0);

    // get pointer to raw data
    float* ptr = output.ptr<float>();

}