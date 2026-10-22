"""Build the four adapters against the pinned Linux MAA release ABI."""

import argparse
import subprocess
from workspace import RELEASE, ROOT, RT

TARGETS = ("maa_rec_input", "maa_ocr_probe", "fd_tensor_probe", "maa_acceptance_probe")


def commands(targets=TARGETS):
    """集中生成编译参数；各目标只声明输出形式和所需链接库。"""
    source = RT / "vendor/MAA-release"
    toolchain = RT / "toolchains/native"
    includes = source / "src/MaaUtils/MaaDeps/vcpkg/installed/maa-x64-linux/include"
    common = [str(toolchain / "bin/clang++"), "-O2", "-std=c++20", "-stdlib=libc++", "-nostdlib++"]
    result = []
    for name in targets:
        if name not in TARGETS:
            raise ValueError("unknown native target: " + name)
        shared = name == "maa_rec_input"
        directories = [toolchain / "include/c++/v1"]
        if not shared:
            directories += [RELEASE, source / "src/MaaCore", source / "src/MaaUtils/include"]
        directories += [includes, includes / "opencv4"]
        headers = ["-I" + str(path) for path in directories]
        libs = [] if shared else ["-lMaaCore", "-lMaaUtils"]
        if name in {"fd_tensor_probe", "maa_acceptance_probe"}:
            libs += ["-l:libonnxruntime.so.1", "-l:libfastdeploy_ppocr.so"]
        elif shared:
            libs += ["-l:libfastdeploy_ppocr.so"]
        libs += ["-l:libopencv_world4.so.412", "-l:libc++.so.1", "-l:libc++abi.so.1",
                 "-l:libunwind.so.1", "-pthread", "-ldl"]
        output = "libmaa_rec_input.so" if shared else name
        result.append(common + headers + (["-shared", "-fPIC"] if shared else []) + [
            str(ROOT / "native" / (name + ".cpp")), "-L" + str(RELEASE),
            "-Wl,-rpath," + str(RELEASE), "-Wl,-rpath-link," + str(RELEASE),
        ] + ([] if shared else ["-Wl,--export-dynamic"]) + libs + ["-o", str(RT / "build" / output)])
    return result


def main():
    """编译全部或明确指定的适配器，失败立即停止。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, action="append")
    args = parser.parse_args()
    (RT / "build").mkdir(parents=True, exist_ok=True)
    for command in commands(args.target or TARGETS):
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
