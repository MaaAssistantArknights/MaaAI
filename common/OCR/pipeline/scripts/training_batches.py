"""Prepare auditable OCR batches and score the fixed dev/guard sets.

The released MAA preprocessor and width grouping stay identical to the former
training entry; only ownership of image loading and inference moves here.
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import paddle
from file_utils import sha
from label_contract import inspect_label
from ppocr.data.imaug.rec_img_aug import resize_norm_img
from released_input import ReleasedInput, width_groups


def prepare_training_batches(
    base, model, encoder, post, input_shape, ctc_steps, kind, batch_size, input_policy
):
    """读取冻结图像并建立训练、开发批次及发布式预处理。"""
    if any(not rows for rows in base.values()):
        raise ValueError("empty selected train/dev/guard split")
    # Input arrays are retained for the finite run. This estimate excludes
    # decoder targets, Python objects and temporary copies during stacking.
    estimated = sum(len(rows) for rows in base.values()) * int(np.prod(input_shape)) * 4
    print(
        json.dumps(
            {
                "event": "memory_estimate",
                "input_tensor_bytes": estimated,
                "scope": "fixed-shape input only; actual process peak can be higher",
            }
        ),
        flush=True,
    )
    datasets = {}
    native_widths = {}
    capacity = {}
    adapter = None
    if input_policy != "fixed":
        adapter = ReleasedInput()
    for split, rows in base.items():
        arrays = []
        native_widths[split] = []
        for row in rows:
            path = Path(row["image"])
            if sha(path) != row["sha256"]:
                raise ValueError("image changed")
            check = inspect_label(
                row["label"],
                encoder,
                ctc_steps,
                literal_ascii=bool(
                    row.get("literal_ascii")
                    or (
                        split == "guard"
                        and kind == "char"
                        and row.get("task") in {"ascii_coverage", "confusable"}
                    )
                ),
            )
            if check["problems"]:
                raise ValueError(check)
            data = encoder({"label": row["label"], "image": None})
            im = cv2.imread(str(path))
            if im is None:
                raise ValueError("cannot decode image")
            norm, ratio = resize_norm_img(im, input_shape)
            if adapter:
                # Released MAA preprocessing can yield different image widths.
                # Check each actual CTC capacity instead of assuming the fixed
                # training width is safe for every sample.
                native = adapter(path)
                width = native.shape[2]
                native_widths[split].append(width)
                if width not in capacity:
                    with paddle.no_grad():
                        capacity[width] = int(
                            model(paddle.zeros([1, 3, 48, width])).shape[1]
                        )
                check = inspect_label(
                    row["label"],
                    encoder,
                    capacity[width],
                    literal_ascii=bool(row.get("literal_ascii")),
                )
                if check["problems"]:
                    raise ValueError({"id": row["id"], "native_width": width, **check})
                norm = native
                ratio = 1.0
            arrays.append(
                [
                    norm,
                    data["label_ctc"],
                    data["label_gtc"],
                    data["length"],
                    np.float32(ratio),
                ]
            )
        datasets[split] = [
            ([r[0] for r in arrays] if adapter else np.stack([r[0] for r in arrays])),
            *[np.stack([r[i] for r in arrays]) for i in range(1, 5)],
        ]
        print(
            json.dumps({"event": "loaded", "split": split, "samples": len(rows)}),
            flush=True,
        )

    def batch(split, idx):
        """按已冻结索引返回一次训练所需的图像和标签。"""
        if not adapter:
            return [paddle.to_tensor(x[idx]) for x in datasets[split]]
        return [
            paddle.to_tensor(np.stack([datasets[split][0][i] for i in idx])),
            *[paddle.to_tensor(x[idx]) for x in datasets[split][1:]],
        ]

    def evaluate(split):
        """用开发分区计算检查点的识别指标。"""
        model.eval()
        pred = []
        with paddle.no_grad():
            if adapter:
                pred = [""] * len(base[split])
                for group in width_groups(
                    range(len(base[split])), native_widths[split]
                ):
                    for off in range(0, len(group), batch_size):
                        idx = group[off : off + batch_size]
                        texts = post(model(batch(split, idx)[0]))
                        for i, v in zip(idx, texts, strict=True):
                            pred[i] = v[0]
            else:
                for off in range(0, len(base[split]), batch_size):
                    pred.extend(
                        v[0]
                        for v in post(
                            model(batch(split, slice(off, off + batch_size))[0])
                        )
                    )
        by = defaultdict(lambda: {"count": 0, "correct": 0})
        wrong = []
        for row, text in zip(base[split], pred, strict=True):
            g = by[row["scene"]]
            g["count"] += 1
            g["correct"] += text == row["label"]
            if text != row["label"]:
                wrong.append(
                    {
                        "id": row["id"],
                        "scene": row["scene"],
                        "label": row["label"],
                        "prediction": text,
                    }
                )
        return {
            "count": len(pred),
            "correct": len(pred) - len(wrong),
            "groups": dict(by),
            "errors": wrong,
        }

    return adapter, native_widths, capacity, batch, evaluate
