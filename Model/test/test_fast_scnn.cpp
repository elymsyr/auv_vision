// g++ -std=c++17 test_fast_scnn.cpp -o test_fast_scnn \
//     -I$CONDA_PREFIX/include \
//     -I$CONDA_PREFIX/include/libtorch/include \
//     -I$CONDA_PREFIX/include/torch/csrc/api/include \
//     -I$CONDA_PREFIX/include/libtorch/include/torch/csrc/api/include \
//     -L$CONDA_PREFIX/lib -ltorch -lc10 \
//     -I$CONDA_PREFIX/include/opencv4 \
//     -L$CONDA_PREFIX/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
//     -ltorch -ltorch_cpu -lc10 \
//     -lpthread -ldl
// LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./test_fast_scnn

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// Depthwise Separable Convolution
struct DepthwiseSeparableConvImpl : torch::nn::Module {
    DepthwiseSeparableConvImpl(int in_channels, int out_channels, int stride = 1) {
        depthwise = register_module("depthwise",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3)
                .stride(stride)
                .padding(1)
                .groups(in_channels)
                .bias(false)));
        pointwise = register_module("pointwise",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                .stride(1)
                .padding(0)
                .bias(false)));
        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = depthwise->forward(x);
        x = pointwise->forward(x);
        x = bn->forward(x);
        return torch::relu(x);  // Functional ReLU
    }
    
    torch::nn::Conv2d depthwise{nullptr};
    torch::nn::Conv2d pointwise{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};
TORCH_MODULE(DepthwiseSeparableConv);

// FastSCNN Model
struct FastSCNNImpl : torch::nn::Module {
    FastSCNNImpl(int n_classes) {
        // Downsample path
        downsample = register_module("downsample", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(2).padding(1).bias(false)),
            torch::nn::BatchNorm2d(32),
            torch::nn::Functional(torch::relu),
            DepthwiseSeparableConv(32, 48, 2),
            DepthwiseSeparableConv(48, 64, 2)
        ));
        
        // Global feature path
        global_feature = register_module("global_feature", torch::nn::Sequential(
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64),
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1))
        ));
        
        // Classifier
        classifier = register_module("classifier",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, n_classes, 1)));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        auto orig_size = x.sizes().vec();
        x = downsample->forward(x);
        auto g = global_feature->forward(x);
        g = g.expand({g.size(0), g.size(1), x.size(2), x.size(3)});  // Expand to match spatial dims
        x = x + g;
        x = classifier->forward(x);
        x = torch::nn::functional::interpolate(x,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({orig_size[2], orig_size[3]}))
                .mode(torch::kBilinear)
                .align_corners(false));
        return x;
    }
    
    torch::nn::Sequential downsample{nullptr};
    torch::nn::Sequential global_feature{nullptr};
    torch::nn::Conv2d classifier{nullptr};
};
TORCH_MODULE(FastSCNN);

int main() {
    // Set device to GPU
    torch::Device device = torch::kCUDA;
    std::cout << "Using CUDA" << std::endl;

    // Create and load model
    FastSCNN model(2);
    try {
        torch::load(model, "fast_scnn_pipe.pth");
        model->to(device);
        model->eval();
        std::cout << "Model loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Load and preprocess image
    cv::Mat img = cv::imread("input0.png");
    if (img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
    
    // Preprocessing
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    // Convert to tensor
    torch::Tensor tensor = torch::from_blob(
        resized.data, {resized.rows, resized.cols, 3}, torch::kByte);
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32).div(255);
    tensor = tensor.to(device);

    // Inference
    torch::Tensor output = model->forward(tensor);
    torch::Tensor pred = output.squeeze(0).argmax(0).detach().cpu();

    // Convert to OpenCV mask
    cv::Mat mask(pred.size(0), pred.size(1), CV_8U);
    std::memcpy(mask.data, pred.data_ptr(), pred.numel() * sizeof(uint8_t));

    // Visualization
    cv::Mat color_mask;
    cv::applyColorMap(mask * 255, color_mask, cv::COLORMAP_JET);
    cv::imshow("Input", img);
    cv::imshow("Segmentation", color_mask);
    cv::waitKey(0);

    return 0;
}