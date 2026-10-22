// Invoke OCR code from the unmodified official MAA shared library.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "AsstCaller.h"
#include "Config/Miscellaneous/OcrPack.h"
#include "Config/Miscellaneous/OcrConfig.h"
#include "Vision/OCRer.h"
#include "Vision/RegionOCRer.h"
using J = nlohmann::json;
using Clock = std::chrono::steady_clock;
// 使用未修改的发布库执行识别或任务规则，并按固定预算记录响应时间。
int main(int argc, char** argv) try {
    if (argc != 8 && argc != 9 && argc != 10) throw std::runtime_error("usage: probe RELEASE PACK manifest.jsonl output.json WARMUP CALLS ROUNDS [CLIENT] [char|word]");
    std::filesystem::path release(argv[1]), pack(argv[2]), output(argv[4]);
    std::filesystem::create_directories(output.parent_path());
    AsstSetUserDir(output.parent_path().c_str());
    if (!asst::OcrConfig::get_instance().load(release / "resource/ocr_config.json")) throw std::runtime_error("ocr_config load failed");
    const std::string kind=argc==10?argv[9]:"char";
    if(kind!="char" && kind!="word") throw std::runtime_error("unknown OCR kind");
    asst::OcrPack& model=kind=="word"?static_cast<asst::OcrPack&>(asst::WordOcr::get_instance()):static_cast<asst::OcrPack&>(asst::CharOcr::get_instance()); model.use_cpu();
    if (!model.load(pack)) throw std::runtime_error("pack load failed");
    std::ifstream input(argv[3]); if (!input) throw std::runtime_error("manifest not found");
    std::vector<J> rows; std::vector<cv::Mat> images;
    for (std::string line; std::getline(input,line);) {
        if (line.empty()) continue;
        auto row=J::parse(line); auto im=cv::imread(row.at("image").get<std::string>());
        if (im.empty()) throw std::runtime_error("unreadable image");
        if(row.contains("resize")) cv::resize(im,im,cv::Size(row["resize"][0],row["resize"][1]));
        rows.push_back(row); images.push_back(im);
    }
    if (rows.empty()) throw std::runtime_error("empty manifest");
    bool tasks=false; for(auto& row:rows) tasks |= row.contains("task_name");
    if(tasks) {
        if(!AsstLoadResource(release.c_str())) throw std::runtime_error("release resources failed to load");
        std::string client=argc>=9?argv[8]:"Official";
        if(client!="Official" && !AsstLoadResource((release/"resource/global"/client).c_str())) throw std::runtime_error("client overlay failed");
        if(!model.load(pack)) throw std::runtime_error("candidate override failed");
    }
    // 区分原始识别、检测识别和任务规则，避免把裁剪诊断当作完整业务验收。
    auto predict=[&](size_t i) {
        const auto& row=rows[i]; const auto& im=images[i];
        asst::Rect roi(0,0,im.cols,im.rows);
        if(row.contains("roi")) roi=asst::Rect(row["roi"][0],row["roi"][1],row["roi"][2],row["roi"][3]);
        auto mode=row.value("mode",std::string("rec"));
        asst::OcrPack::ResultsVec result;
        if(mode=="task_region") {
            asst::RegionOCRer o(im); o.set_task_info(row.at("task_name").get<std::string>());
            if(auto r=o.analyze()) result.push_back(*r);
        }
        else if(mode=="task_crop") {
            // Diagnostic: keep released task text rules, but use a supplied crop.
            // This excludes full-frame detection, dynamic ROI, and business decisions.
            asst::OCRer o(im); o.set_task_info(row.at("task_name").get<std::string>());
            o.set_roi(roi); o.set_without_det(true); o.set_use_char_model(kind=="char");
            if(row.contains("runtime_required")) o.set_required(row["runtime_required"].get<std::vector<std::string>>());
            if(auto r=o.analyze()) result=*r;
        }
        else if(mode=="rec" || mode=="det_rec") result=model.recognize(im(cv::Rect(roi.x,roi.y,roi.width,roi.height)),mode=="rec",roi);
        else {
            asst::OCRerConfig::Params p; p.use_char_model=kind=="char"; p.without_det=true;
            auto cfg=row.value("params",J::object());
            p.bin_threshold_lower=cfg.value("bin_threshold_lower",140); p.bin_threshold_upper=cfg.value("bin_threshold_upper",255);
            p.bin_expansion=cfg.value("bin_expansion",2); p.bin_left_trim_threshold=cfg.value("bin_left_trim_threshold",0); p.bin_right_trim_threshold=cfg.value("bin_right_trim_threshold",0);
            p.use_raw=cfg.value("use_raw",true); p.full_match=cfg.value("full_match",false); p.replace_full=cfg.value("replace_full",false);
            if(cfg.contains("replace")) p.replace=cfg["replace"].get<decltype(p.replace)>();
            if(cfg.contains("required")) for(auto& s:cfg["required"]) { auto text=s.get<std::string>(); p.required.emplace_back(text,asst::OcrConfig::get_instance().process_equivalence_class(text)); }
            if(mode=="region") { asst::RegionOCRer o(im,roi); o.set_params(p); if(auto r=o.analyze()) result.push_back(*r); }
            else if(mode=="ocr") { asst::OCRer o(im,roi); o.set_params(p); if(auto r=o.analyze()) result=*r; }
            else throw std::runtime_error("unknown mode");
        }
        return result;
    };
    auto start=Clock::now(); auto first=predict(0);
    double cold_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    J predictions=J::array();
    for(size_t i=0;i<rows.size();++i) {
        auto r=predict(i); J decoded=J::array();
        for(auto& x:r) decoded.push_back({{"text",x.text},{"score",x.score},{"rect",{x.rect.x,x.rect.y,x.rect.width,x.rect.height}}});
        predictions.push_back({{"id",rows[i].at("id")},{"results",decoded},{"rejected",r.empty()}});
    }
    int warm=std::stoi(argv[5]), calls=std::stoi(argv[6]), rounds=std::stoi(argv[7]);
    if(warm<0 || calls<0 || rounds<0) throw std::runtime_error("negative benchmark budget");
    J timing=J::array();
    for(int round=0;round<rounds;++round) {
        for(int k=0;k<warm;++k) predict(k%rows.size());
        J samples=J::array();
        for(int k=0;k<calls;++k) { size_t i=k%rows.size(); auto t=Clock::now(); auto r=predict(i); double ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count(); samples.push_back({{"id",rows[i]["id"]},{"ms",ms},{"rejected",r.empty()}}); }
        timing.push_back(samples);
    }
    int logical=std::max(1u,std::thread::hardware_concurrency());
    J report={{"client",argc>=9?argv[8]:"Official"},{"implementation","official libMaaCore "+release.filename().string()+"; OcrPack/OCRer/RegionOCRer"},{"ocr_kind",kind},{"pack",pack.string()},{"cpu_threads",logical<=2?1:logical<=4?2:logical<=12?3:4},{"first_predict_ms",cold_ms},{"cold_definition","includes lazy det+rec initialization; excludes process startup and image decoding"},{"predictions",predictions},{"timing",timing}};
    std::ofstream(output)<<report.dump(2)<<'\n';
    std::cout<<"wrote "<<output<<" with "<<rows.size()<<" predictions\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
