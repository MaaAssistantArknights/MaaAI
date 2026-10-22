"""Finite paired replay and timing on the pinned published MAA CPU interface."""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from acceptance_cases import compare_cases
from evaluation_policy import evaluation_failed
from case_io import read, write_rows, manifest
from checkpoint_select import assess
from doctor import inspect
from routes import CLIENT
from file_utils import sha
from workspace import ROOT, RT
from file_utils import acquire_training_lock, atomic_json
from workspace import RELEASE

FILES = ['rec/inference.onnx', 'rec/keys.txt', 'det/inference.onnx']


def fingerprints(pack):
    """计算候选包中模型、字典和检测器的逐文件哈希。"""
    return {name: sha(pack / name) for name in FILES}


def official_pack(route, output):
    """按语言覆盖规则定位官方发布资源并建立只读引用。"""
    kind = 'PaddleCharOCR' if route == 'char' else 'PaddleOCR'
    base = RELEASE / 'resource' / kind
    overlay = RELEASE / 'resource/global' / CLIENT[route] / 'resource' / kind
    for name in FILES:
        src = overlay / name if CLIENT[route] != 'Official' and (overlay / name).exists() else base / name
        dst = output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
    return output


def invoke(command, log, timeout=900):
    """执行原生探针并把输出完整写入独立日志。"""
    with log.open('x') as f:
        subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def summarize_timing(records, expected_ids, calls):
    """在相同负载与线程设置下计算配对 P95 响应时间比率。"""
    p95 = {}
    threads = set()
    for arm in ['official', 'candidate']:
        values = []
        for doc in records[arm]:
            threads.add(doc['cpu_threads'])
            if len(doc['timing']) != 1:
                raise ValueError('unexpected timing rounds')
            samples = doc['timing'][0]
            if [s['id'] for s in samples] != [expected_ids[i % len(expected_ids)] for i in range(calls)]:
                raise ValueError('timing workload changed')
            times = np.asarray([s['ms'] for s in samples])
            if not np.isfinite(times).all() or (times <= 0).any():
                raise ValueError('invalid latency')
            values.append(float(np.quantile(times, .95, method='linear')))
        if not values:
            raise ValueError('missing timing round')
        p95[arm] = {'rounds': values, 'median_ms': statistics.median(values)}
    if len(threads) != 1 or len(records['official']) != len(records['candidate']):
        raise ValueError('unpaired timing settings')
    ratio = p95['candidate']['median_ms'] / p95['official']['median_ms']
    return {'p95': p95, 'response_time_ratio': ratio, 'within_1_05': ratio <= 1.05, 'threads': threads.pop()}


def benchmark(output, route, packs, mp, budget):
    """对官方版与候选版轮换执行冷、热进程计时。"""
    output.mkdir()
    cases = read(mp)
    records = {'official': [], 'candidate': []}
    cold = {'official': [], 'candidate': []}
    cold_mp = output / 'cold.jsonl'
    write_rows(cold_mp, [cases[len(cases) // 2]])
    for phase, count in [('warm', budget['rounds']), ('cold', budget['cold_processes'])]:
        for i in range(count):
            for name in (['official', 'candidate'] if i % 2 == 0 else ['candidate', 'official']):
                dest = output / f'{phase}-{i}-{name}.json'
                rss = dest.with_suffix('.rss')
                args = [RT/'build/maa_ocr_probe', RELEASE, packs[name], mp if phase == 'warm' else cold_mp, dest,
                        budget['warmups'] if phase == 'warm' else 0, budget['calls'] if phase == 'warm' else 0,
                        1 if phase == 'warm' else 0, CLIENT[route], 'char' if route == 'char' else 'word']
                start = time.monotonic()
                invoke(['/usr/bin/time', '-f', '%M', '-o', rss, *args], dest.with_suffix('.log'))
                doc = json.loads(dest.read_text())
                extra = {'first_predict_ms': doc['first_predict_ms'], 'process_wall_seconds': time.monotonic()-start,
                         'peak_rss_kib': int(rss.read_text()), 'output': str(dest)}
                if phase == 'warm':
                    records[name].append(doc)
                else:
                    cold[name].append(extra)
    return {**summarize_timing(records, [r['id'] for r in cases], budget['calls']), 'budget': budget,
            'cold': cold, 'manifest_sha256': sha(mp), 'first_predict_definition': 'native first prediction; excludes process/resource/image loading',
            'wall_definition': 'whole probe process including two single-input predictions, not isolated model loading',
            'scope': 'fixed workload on this host, not Windows or workload-frequency evidence'}


def main():
    """校验固定发布环境并执行识别、整帧与计时评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    cfg = json.loads(a.spec.read_text())
    def path(s):
        """相对评估 spec 解析输入路径，保持同一工作目录语义。"""
        p = Path(s)
        return (p if p.is_absolute() else a.spec.parent/p).resolve()
    if cfg.get('purpose') not in ['integration', 'evaluation']:
        raise ValueError('declare purpose')
    timing_policy = cfg.get('timing_policy', 'record_only')
    if timing_policy not in {'record_only', 'enforce'}:
        raise ValueError('unknown timing policy')
    if not any(cfg.get(k) for k in ['frames', 'recognition', 'timing']):
        raise ValueError('no validation workload')
    budget = cfg.get('budget', {'rounds': 3, 'calls': 1000, 'warmups': 30, 'cold_processes': 10})
    if not (1 <= budget['rounds'] <= 5 and 1 <= budget['calls'] <= 10000 and 0 <= budget['warmups'] <= 1000 and 0 <= budget['cold_processes'] <= 20):
        raise ValueError('timing budget outside bounds')
    if cfg['purpose'] == 'evaluation' and cfg.get('timing') and budget != {'rounds': 3, 'calls': 1000, 'warmups': 30, 'cold_processes': 10}:
        raise ValueError('evaluation requires the fixed timing protocol')
    if not inspect(ROOT/'configs/runtime.lock.json')['passed']:
        raise ValueError('pinned release differs')
    lock = acquire_training_lock(RT)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = {str(p.resolve()): sha(p) for p in [a.spec, *RELEASE.rglob('*.so*')] if p.is_file()}
    # Task JSON and templates are part of the baseline, not just model binaries.
    files.update({str(p): sha(p) for p in (RELEASE/'resource').rglob('*') if p.is_file()})
    files.update({str(p): sha(p) for p in (ROOT/'scripts').glob('*.py')})
    for key in ['frames', 'recognition', 'timing']:
        sources = [cfg[key]] if key == 'frames' and cfg.get(key) else list(cfg.get(key, {}).values()) if key != 'frames' else []
        for source in sources:
            mp = path(source)
            files[str(mp)] = sha(mp)
            for row in read(mp):
                im = Path(row['image'])
                im = (im if im.is_absolute() else mp.parent/im).resolve()
                if sha(im) != row['sha256']:raise ValueError('input image changed')
                files[str(im)] = row['sha256']
    for probe in ['maa_acceptance_probe', 'maa_ocr_probe']:
        f=RT/'build'/probe
        if f.exists():files[str(f)]=sha(f)
    packs = {}
    for route, value in cfg['packs'].items():
        if route not in CLIENT:
            raise ValueError('unknown route')
        packs[route] = {'official': official_pack(route, out/'official'/route), 'candidate': path(value)}
        fp = {n: fingerprints(p) for n, p in packs[route].items()}
        if fp['official']['det/inference.onnx'] != fp['candidate']['det/inference.onnx']:
            raise ValueError('recognition comparison cannot change detector')
        files.update({str(p/name): sha(p/name) for p in packs[route].values() for name in FILES})
    summary = {'purpose': cfg['purpose'], 'release_approved': False, 'training_started': False,
               'platform': platform.platform(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'frames': None, 'recognition': {}, 'timing': {}, 'external_validation_complete': False}
    atomic_json(out/'inputs.json', files)
    try:
        if cfg.get('frames'):
            cases = manifest(path(cfg['frames']), out/'frames.jsonl')
            preds = {'official': [], 'candidate': []}
            for route in sorted({r['route'] for r in cases}):
                mp = out/(route+'-frames.jsonl')
                write_rows(mp, [r for r in cases if r['route'] == route])
                for arm in preds:
                    d = out/f'frames-{route}-{arm}'; d.mkdir()
                    invoke([RT/'build/maa_acceptance_probe', RELEASE, CLIENT[route], packs['char'][arm], packs[route][arm], mp, d/'predictions.json'], d/'probe.log')
                    preds[arm] += json.loads((d/'predictions.json').read_text())['predictions']
            summary['frames'] = compare_cases(cases, preds['official'], preds['candidate'])
        for route, source in cfg.get('recognition', {}).items():
            mp = out/(route+'-recognition.jsonl'); rows = manifest(path(source), mp); preds = {}
            for arm in ['official', 'candidate']:
                d = out/f'recognition-{route}-{arm}'; d.mkdir()
                invoke([RT/'build/maa_ocr_probe', RELEASE, packs[route][arm], mp, d/'predictions.json', 0, 0, 0, CLIENT[route], 'char' if route == 'char' else 'word'], d/'probe.log')
                preds[arm] = json.loads((d/'predictions.json').read_text())['predictions']
            # Both reference arguments are official in this paired baseline comparison.
            summary['recognition'][route] = assess(rows, preds['candidate'], preds['official'], preds['official'])
        for route, source in cfg.get('timing', {}).items():
            mp = out/(route+'-timing.jsonl'); manifest(path(source), mp)
            summary['timing'][route] = benchmark(out/('timing-'+route), route, packs[route], mp, budget)
        for file, digest in files.items():
            if sha(Path(file)) != digest:
                raise ValueError('frozen evaluation dependency changed: '+file)
        summary['state'] = 'completed'
        # Preserve the paired P95 measurement even when this release records
        # latency for future optimization instead of treating it as a veto.
        summary['timing_policy'] = timing_policy
        summary['timing_gate_applicable'] = cfg['purpose'] == 'evaluation' and bool(summary['timing']) and timing_policy == 'enforce'
        summary['failed_timing_routes'] = [r for r,v in summary['timing'].items() if not v['within_1_05']]
        atomic_json(out/'summary.json', summary)
        print('Completed fixed validation; release approval remains separate.')
        if evaluation_failed(cfg, summary):
            raise SystemExit(1)
    except Exception as e:
        atomic_json(out/'status.json', {'state': 'failed', 'error': repr(e), 'release_approved': False})
        raise
    finally:
        lock.close()

if __name__ == '__main__':
    main()
