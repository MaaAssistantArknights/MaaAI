// Diagnostic only: dump tensors produced by the released FastDeploy + ORT libraries.
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision/ocr/ppocr/recognizer.h"
using J=nlohmann::json;
// 导出发布运行库实际产生的输入和输出张量，供转换一致性逐项比较。
int main(int argc,char** argv) try {
 if(argc!=5)throw std::runtime_error("usage: probe MODEL KEYS manifest.jsonl OUTDIR");
 std::filesystem::path out(argv[4]);std::filesystem::create_directories(out);
 fastdeploy::RuntimeOption option;option.UseOrtBackend();option.UseCpu();option.SetCpuThreadNum(3);
 fastdeploy::vision::ocr::Recognizer rec(argv[1],"",argv[2],option,fastdeploy::ModelFormat::ONNX);
 if(!rec.Initialized())throw std::runtime_error("model init failed");
 std::ifstream f(argv[3]);J rows=J::array();int index=0;
 for(std::string line;std::getline(f,line);){
  if(line.empty())continue;J r=J::parse(line);auto im=cv::imread(r.at("image").get<std::string>());
  if(im.empty())throw std::runtime_error("bad image");
  std::vector<fastdeploy::vision::FDMat> images{fastdeploy::vision::FDMat(im)};
  std::vector<fastdeploy::FDTensor> inputs,outputs;
  if(!rec.GetPreprocessor().Run(&images,&inputs))throw std::runtime_error("preprocess failed");
  inputs[0].name=rec.InputInfoOfRuntime(0).name;
  if(!rec.Infer(inputs,&outputs))throw std::runtime_error("infer failed");
  std::string text;float score=0; if(!rec.Predict(im,&text,&score))throw std::runtime_error("predict failed");
  // 按原始浮点字节保存张量，形状与文件大小一并写入清单。
  auto dump=[&](const auto& tensor,const std::string& name){auto path=out/name;std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(tensor.CpuData()),tensor.Nbytes());return J{{"path",path.string()},{"shape",tensor.shape},{"bytes",tensor.Nbytes()}};};
  rows.push_back({{"id",r["id"]},{"text",text},{"score",score},{"input",dump(inputs[0],std::to_string(index)+"-input.bin")},{"output",dump(outputs[0],std::to_string(index)+"-output.bin")}});++index;
 }
 std::ofstream(out/"manifest.json")<<rows.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
