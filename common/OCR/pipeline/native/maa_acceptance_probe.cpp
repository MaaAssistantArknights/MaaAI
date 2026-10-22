// Whole-frame regression checks call unchanged released analyzers. No controller/actions.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "AsstCaller.h"
#include "Config/Miscellaneous/OcrPack.h"
#include "Vision/RegionOCRer.h"
#include "Vision/Miscellaneous/RecruitImageAnalyzer.h"
#include "Vision/Miscellaneous/StageDropsImageAnalyzer.h"
#include "Vision/Miscellaneous/DepotImageAnalyzer.h"
#include "Vision/Miscellaneous/CreditShopImageAnalyzer.h"
using J=nlohmann::json;
// 将发布分析器的区域坐标转为稳定的 JSON 字段。
J rect(const asst::Rect& r){return J::array({r.x,r.y,r.width,r.height});}
// 保留文字、区域和置信分数，供业务比较按各自规则判定。
J texts(const auto& results){J out=J::array();for(const auto& r:results)out.push_back({{"text",r.text},{"rect",rect(r.rect)},{"score",r.score}});return out;}
// 对整帧图片调用发布版分析器，只读取结果，不执行游戏操作。
int main(int argc,char** argv)try{
 if(argc!=7)throw std::runtime_error("usage: RELEASE CLIENT CHAR_PACK WORD_PACK manifest.jsonl output.json");
 std::filesystem::path root(argv[1]),out(argv[6]);std::filesystem::create_directories(out.parent_path());AsstSetUserDir(out.parent_path().c_str());
 if(!AsstLoadResource(root.c_str()))throw std::runtime_error("base resource load failed");
 const std::string client(argv[2]);
 if(client!="Official"&&!AsstLoadResource((root/"resource/global"/client).c_str()))throw std::runtime_error("client load failed");
 auto& chr=asst::CharOcr::get_instance();auto& word=asst::WordOcr::get_instance();chr.use_cpu();word.use_cpu();
 if(!chr.load(argv[3])||!word.load(argv[4]))throw std::runtime_error("pack load failed");
 std::ifstream input(argv[5]);if(!input)throw std::runtime_error("manifest missing");J rows=J::array();
 for(std::string line;std::getline(input,line);){
  if(line.empty())continue;auto row=J::parse(line);auto im=cv::imread(row.at("image").get<std::string>());if(im.empty())throw std::runtime_error("image missing");
  if(row.contains("resize"))cv::resize(im,im,cv::Size(row["resize"][0],row["resize"][1]));
  if(im.cols!=1280||im.rows!=720)throw std::runtime_error("1280x720 MAA frame required");
  const auto mode=row.at("mode").get<std::string>();J result={{"id",row.at("id")},{"mode",mode}};
  // Mark frame boundaries so fallback evidence can be read from original MAA logs.
  std::cout<<"FRAME_BEGIN "<<row.at("id").get<std::string>()<<std::endl;
  if(mode=="stage_drops"){
   asst::StageDropsImageAnalyzer a(im);result["analyze_ok"]=a.analyze();auto key=a.get_stage_key();
   result["stage_code"]=key.code;result["difficulty"]=asst::enum_to_string(key.difficulty);result["stars"]=a.get_stars();result["times"]=a.get_times();result["drops"]=J::array();
   for(const auto& d:a.get_drops())result["drops"].push_back({{"type",static_cast<int>(d.drop_type)},{"item_id",d.item_id},{"quantity",d.quantity}});
  }else if(mode=="recruit"){
   asst::RecruitImageAnalyzer a(im);result["analyze_ok"]=a.analyze();result["results"]=texts(a.get_tags_result());
   result["hour_button"]=rect(a.get_hour_decrement_rect());result["minute_button"]=rect(a.get_minute_decrement_rect());result["refresh_button"]=rect(a.get_refresh_rect());result["no_permit_indicator"]=rect(a.get_permit_rect());
  }else if(mode=="credit_shop"){
   asst::CreditShopImageAnalyzer a(im);auto list=row.value("shopping_list",std::vector<std::string>{});
   if(row.value("list_mode",std::string("black"))=="white")a.set_white_list(list);else a.set_black_list(list);
   result["analyze_ok"]=a.analyze();result["results"]=texts(a.get_result());
  }else if(mode=="depot"){
   asst::DepotImageAnalyzer a(im);result["analyze_ok"]=a.analyze();result["results"]=J::array();
   for(const auto& [id,v]:a.get_result())result["results"].push_back({{"item_id",id},{"quantity",v.quantity},{"rect",rect(v.rect)}});
  }else if(mode=="task_region"){
   asst::RegionOCRer a(im);a.set_task_info(row.at("task_name").get<std::string>());auto r=a.analyze();result["analyze_ok"]=static_cast<bool>(r);result["results"]=J::array();if(r)result["results"].push_back({{"text",r->text},{"rect",rect(r->rect)},{"score",r->score}});
  }else throw std::runtime_error("unsupported mode");
  std::cout<<"FRAME_END "<<row.at("id").get<std::string>()<<std::endl;rows.push_back(result);
 }
 std::ofstream(out)<<J{{"release",root.filename().string()},{"client",client},{"char_pack",argv[3]},{"word_pack",argv[4]},{"predictions",rows},{"actions_executed",false},{"live_workflow_tested",false}}.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
