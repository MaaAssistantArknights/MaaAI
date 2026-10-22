"""Inspect the chosen training GPU and versioned resource limits."""

import json
import shutil
import subprocess
import uuid
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1] / "configs/resource-policy.json"
FIELDS = ("index", "uuid", "used_mib", "free_mib", "temperature_c", "utilization_percent")


def load_policy(path=POLICY):
    """加载单卡训练的显存、磁盘和温度门槛。"""
    policy = json.loads(Path(path).read_text())
    if (policy.get("version") != 2 or type(policy.get("gpu_index")) is not int
            or policy["gpu_index"] < 0):
        raise ValueError("unknown GPU resource policy")
    for field in ("maximum_existing_gpu_memory_mib", "minimum_free_gpu_memory_mib",
                  "maximum_gpu_temperature_c", "minimum_free_runtime_disk_gib"):
        value = policy.get(field)
        if type(value) not in (int, float):
            raise ValueError("invalid resource limit: " + field)
        minimum_ok = value >= 0 if field == "maximum_existing_gpu_memory_mib" else value > 0
        if not minimum_ok or not value < float("inf"):
            raise ValueError("invalid resource limit: " + field)
    return policy


def cuda_device_uuid(device):
    """从 CUDA 的逻辑设备取得 UUID，使监测与 Paddle 的可见设备顺序一致。"""
    from cuda.bindings import runtime

    # Let CUDA resolve CUDA_VISIBLE_DEVICES, UUID selectors and container remapping.
    # Interpreting those strings as nvidia-smi indices can inspect another GPU.
    error, properties = runtime.cudaGetDeviceProperties(device)
    if int(error) != 0:
        raise RuntimeError(f"CUDA device gpu:{device} is unavailable: {error}")
    return "GPU-" + str(uuid.UUID(bytes=properties.uuid.bytes))


def gpu_status(policy=None):
    """查询训练 GPU 的身份、空闲显存和温度。"""
    policy = policy or load_policy()
    device = policy["gpu_index"]  # CUDA-visible ordinal, also passed to paddle.set_device.
    selected_uuid = cuda_device_uuid(device)
    command = [
        "nvidia-smi", f"--id={selected_uuid}",
        "--query-gpu=index,uuid,memory.used,memory.free,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    rows = output.splitlines()
    if len(rows) != 1:
        raise RuntimeError("expected exactly one selected GPU from nvidia-smi")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(FIELDS):
        raise RuntimeError("unexpected nvidia-smi field count")
    result = dict(zip(FIELDS, values, strict=True))
    for key in FIELDS:
        if key != "uuid":
            result[key] = int(result[key])
    if result["uuid"] != selected_uuid:
        raise RuntimeError("GPU query returned a different device UUID")
    result["cuda_index"] = device
    return result


def require_training_capacity(runtime, policy=None):
    """训练前检查工作目录空间和 GPU 可用容量。"""
    policy = policy or load_policy()
    status = gpu_status(policy)
    if (status["used_mib"] > policy["maximum_existing_gpu_memory_mib"] or
            status["free_mib"] < policy["minimum_free_gpu_memory_mib"]):
        raise RuntimeError("selected GPU does not meet the training memory policy")
    if shutil.disk_usage(runtime).free < policy["minimum_free_runtime_disk_gib"] * 1024**3:
        raise RuntimeError("runtime disk does not meet the training space policy")
    require_safe_temperature(status, policy)
    return status


def require_safe_temperature(status, policy=None):
    """超出资源策略温度时阻止继续训练。"""
    policy = policy or load_policy()
    if status["temperature_c"] >= policy["maximum_gpu_temperature_c"]:
        raise RuntimeError("selected GPU exceeds the temperature policy")
