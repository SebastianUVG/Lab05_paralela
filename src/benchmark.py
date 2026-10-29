"""Guarda mediciones crudas y metadatos; no genera el informe."""

import argparse
import csv
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from numba import config, cuda

from .cuda_impl import THREADS, run_cuda
from .data import PERFORMANCE_CASES
from .sequential import run_sequential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials debe ser positivo")
    if config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Las mediciones requieren GPU real y CUDA disponible")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output or Path("results") / stamp
    folder.mkdir(parents=True, exist_ok=False)
    device = cuda.get_current_device()
    metadata = {
        "started_utc": stamp, "python": platform.python_version(),
        "platform": platform.platform(), "gpu": str(device.name),
        "compute_capability": device.compute_capability,
        "threads_per_block": THREADS, "trials_per_size": args.trials,
        "rtol": 1e-11, "atol": 1e-12, "simulator": False,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "numba", "numba-cuda", "cuda-bindings")},
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("src/data.py", "src/sequential.py",
                                       "src/cuda_impl.py", "src/benchmark.py")},
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "method": "Una ejecución CPU y GPU de calentamiento por tamaño, excluidas. "
                  "Después CPU seguida de GPU por repetición. Total CUDA incluye "
                  "reservas, H2D, todos los kernels de reducción y D2H; excluye "
                  "contexto, JIT, preparación de parámetros y liberación posterior. "
                  "Transferencias de entrada: 24 bytes; salida: 8 bytes. "
                  "Cronómetro de pared perf_counter, GPU sincronizada."
    }
    (folder / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fields = ["n", "a", "b", "trial", "cpu_seconds", "allocation_seconds",
              "h2d_seconds", "kernel_seconds", "d2h_seconds", "cuda_total_seconds",
              "device_bytes", "cpu_result", "gpu_result", "absolute_error",
              "correct", "speedup"]
    with (folder / "measurements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case in PERFORMANCE_CASES:
            run_sequential(case)
            run_cuda(case)
            for trial in range(1, args.trials + 1):
                start = perf_counter()
                cpu = run_sequential(case)
                cpu_seconds = perf_counter() - start
                gpu = run_cuda(case)
                correct = math.isclose(cpu, gpu.result, rel_tol=1e-11, abs_tol=1e-12)
                row = asdict(gpu)
                row["gpu_result"] = row.pop("result")
                row.update(n=case.n, a=case.a, b=case.b, trial=trial,
                           cpu_seconds=cpu_seconds, cpu_result=cpu,
                           absolute_error=abs(cpu - gpu.result), correct=correct,
                           speedup=cpu_seconds / gpu.cuda_total_seconds)
                writer.writerow(row)
                file.flush()
                if not correct:
                    raise AssertionError(f"CPU/GPU difieren: {case}, repetición {trial}")
            print(f"n={case.n}: {args.trials} corridas guardadas", flush=True)
    print(folder)


if __name__ == "__main__":
    main()
