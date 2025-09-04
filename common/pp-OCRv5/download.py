#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, shutil, subprocess, zipfile, urllib.request, tempfile, glob

REPO = "Kengxxiao/ArknightsGameData"
REPO_URL = f"https://github.com/{REPO}.git"
ZIP_CANDIDATES = [
    f"https://codeload.github.com/{REPO}/zip/refs/heads/main",
    f"https://codeload.github.com/{REPO}/zip/refs/heads/master",
]

FONT_BASE = "https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R"
# 可通过命令行传参：python setup_fonts.py CN
font_lang = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("fontLang", "CN")

def run(cmd, cwd=None, check=True):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)

def have_cmd(name):
    return shutil.which(name) is not None

def ensure_repo():
    if os.path.isdir("ArknightsGameData/.git"):
        # 已是 git 仓库，尝试拉取
        if have_cmd("git"):
            try:
                run(["git", "-C", "ArknightsGameData", "pull", "--ff-only"])
                return
            except subprocess.CalledProcessError:
                print("git pull 失败，将回退到下载 zip 的方式")
        else:
            print("未找到 git，可直接使用现有目录（不会更新），或删除后走 zip 下载。")
            return

    if os.path.isdir("ArknightsGameData") and not os.path.isdir("ArknightsGameData/.git"):
        print("检测到同名非 git 目录，将使用该目录（不更新）。如需更新，请删除后重试。")
        return

    # 没目录或不是git时：优先 git clone，失败再 zip
    if have_cmd("git"):
        try:
            run(["git", "clone", "--depth=1", REPO_URL, "ArknightsGameData"])
            return
        except subprocess.CalledProcessError:
            print("git clone 失败，尝试下载 zip。")

    # 下载仓库 zip（main 或 master）
    os.makedirs("ArknightsGameData", exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for url in ZIP_CANDIDATES:
            try:
                print(f"下载仓库：{url}")
                zip_path = os.path.join(td, "repo.zip")
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    # 解压到临时目录，再把内容移动到 ArknightsGameData
                    zf.extractall(td)
                # 找到唯一的根目录
                roots = [p for p in os.listdir(td) if os.path.isdir(os.path.join(td, p)) and p.lower().startswith("arknightsgamedata-")]
                if not roots:
                    continue
                src = os.path.join(td, roots[0])
                # 清空目标目录
                for name in os.listdir("ArknightsGameData"):
                    path = os.path.join("ArknightsGameData", name)
                    if os.path.isfile(path) or os.path.islink(path):
                        os.unlink(path)
                    else:
                        shutil.rmtree(path)
                # 移动文件
                for name in os.listdir(src):
                    shutil.move(os.path.join(src, name), os.path.join("ArknightsGameData", name))
                print("仓库内容已通过 zip 就绪。")
                return
            except Exception as e:
                print(f"下载/解压失败：{e}")
        print("无法通过 zip 获取仓库，请检查网络。")

def download_fonts():
    fonts_dir = "fonts"
    os.makedirs(fonts_dir, exist_ok=True)
    font_zip = os.path.join(fonts_dir, f"SourceHanSans{font_lang}.zip")
    url = f"{FONT_BASE}/SourceHanSans{font_lang}.zip"
    if not os.path.exists(font_zip):
        print(f"下载字体：{url}")
        urllib.request.urlretrieve(url, font_zip)
    else:
        print(f"已存在字体包：{font_zip}（跳过下载）")

    print(f"解压字体到 {fonts_dir}/")
    with zipfile.ZipFile(font_zip) as zf:
        zf.extractall(fonts_dir)

    subset_path = os.path.join(fonts_dir, "SubsetOTF", font_lang)
    if not os.path.isdir(subset_path):
        raise SystemExit(f"未找到子集目录：{subset_path}，请检查 fontLang='{font_lang}' 是否正确。")

    # 生成 fonts.txt（绝对路径）
    font_files = sorted(glob.glob(os.path.join(subset_path, "*")))
    if not font_files:
        raise SystemExit(f"未在 {subset_path} 找到字体文件。")
    fonts_txt = os.path.join(fonts_dir, "fonts.txt")
    with open(fonts_txt, "w", encoding="utf-8") as f:
        for p in font_files:
            f.write(os.path.abspath(p) + "\n")
    print(f"已生成 {fonts_txt}，共 {len(font_files)} 条。")

def main():
    # 切到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("注意请 cd 到当前目录下再运行（本脚本会自动切换到脚本目录）")
    print(f"fontLang: {font_lang}")

    ensure_repo()
    download_fonts()
    print("全部完成")

if __name__ == "__main__":
    main()
