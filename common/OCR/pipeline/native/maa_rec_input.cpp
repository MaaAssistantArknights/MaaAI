// Training adapter only: execute the pinned release's unmodified preprocessing.
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision/ocr/ppocr/rec_preprocessor.h"
static thread_local std::string error;
// 返回当前线程最近一次预处理失败的说明，由调用方只读。
extern "C" const char* maa_input_error() { return error.c_str(); }
// 释放本库分配的张量内存；Python 必须先复制数据再调用此函数。
extern "C" void maa_input_free(float* p) { std::free(p); }
// 使用固定发布版的预处理器；成功时返回拥有独立内存的 CHW 张量。
extern "C" int maa_rec_input(const char* path, float** data, int* height, int* width) {
    *data = nullptr;
    try {
        cv::setNumThreads(1);
        auto im = cv::imread(path);
        if (im.empty()) throw std::runtime_error("cannot read image");
        fastdeploy::vision::ocr::RecognizerPreprocessor pre;
        std::vector<fastdeploy::vision::FDMat> images{fastdeploy::vision::FDMat(im)};
        std::vector<fastdeploy::FDTensor> outputs;
        if (!pre.Run(&images, &outputs)) throw std::runtime_error("released preprocessing failed");
        auto& t = outputs.at(0);
        if (t.shape.size() != 4 || t.shape[0] != 1 || t.shape[1] != 3 || t.shape[2] != 48)
            throw std::runtime_error("unexpected released input shape");
        *height = t.shape[2]; *width = t.shape[3];
        *data = static_cast<float*>(std::malloc(t.Nbytes()));
        if (!*data) throw std::bad_alloc();
        std::memcpy(*data, t.CpuData(), t.Nbytes());
        return 0;
    } catch (const std::exception& e) { error=e.what(); return 1; }
}
