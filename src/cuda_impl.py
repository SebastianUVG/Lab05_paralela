"""Punto medio float64 con reducción completa en GPU."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import cuda, float64

from .data import IntegrationInput

THREADS = 256


@cuda.jit
def _integrate(bounds, count, output):
    shared = cuda.shared.array(THREADS, float64)
    i = cuda.grid(1)
    t = cuda.threadIdx.x
    n = count[0]
    value = 0.0
    if i < n:
        width = (bounds[1] - bounds[0]) / n
        x = bounds[0] + (i + 0.5) * width
        value = (4.0 / (1.0 + x * x)) * width
    shared[t] = value
    cuda.syncthreads()
    stride = THREADS // 2
    while stride > 0:
        if t < stride:
            shared[t] += shared[t + stride]
        cuda.syncthreads()
        stride //= 2
    if t == 0:
        output[cuda.blockIdx.x] = shared[0]


@cuda.jit
def _reduce(values, output):
    shared = cuda.shared.array(THREADS, float64)
    i = cuda.grid(1)
    t = cuda.threadIdx.x
    value = 0.0
    if i < values.size:
        value = values[i]
    shared[t] = value
    cuda.syncthreads()
    stride = THREADS // 2
    while stride > 0:
        if t < stride:
            shared[t] += shared[t + stride]
        cuda.syncthreads()
        stride //= 2
    if t == 0:
        output[cuda.blockIdx.x] = shared[0]


@dataclass(frozen=True)
class CudaRun:
    result: float
    allocation_seconds: float
    h2d_seconds: float
    kernel_seconds: float
    d2h_seconds: float
    cuda_total_seconds: float
    device_bytes: int


def run_cuda(inputs: IntegrationInput) -> CudaRun:
    """Mide una ejecución; el llamador debe calentar CUDA antes del benchmark.

    Total de pared incluye reservas, copias, kernels y coordinación Python.
    Preparación de parámetros e inicialización de contexto quedan fuera.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA no está disponible; se requiere una GPU real")
    bounds = np.array([inputs.a, inputs.b], dtype=np.float64)
    count = np.array([inputs.n], dtype=np.int64)
    host_result = np.empty(1, dtype=np.float64)
    cuda.synchronize()
    start = perf_counter()
    d_bounds = cuda.device_array(2, dtype=np.float64)
    d_count = cuda.device_array(1, dtype=np.int64)
    size = (inputs.n + THREADS - 1) // THREADS
    buffers = [cuda.device_array(size, dtype=np.float64)]
    while size > 1:
        size = (size + THREADS - 1) // THREADS
        buffers.append(cuda.device_array(size, dtype=np.float64))
    allocated = perf_counter()
    d_bounds.copy_to_device(bounds)
    d_count.copy_to_device(count)
    cuda.synchronize()
    copied = perf_counter()
    _integrate[buffers[0].size, THREADS](d_bounds, d_count, buffers[0])
    for previous, following in zip(buffers, buffers[1:]):
        _reduce[following.size, THREADS](previous, following)
    cuda.synchronize()
    calculated = perf_counter()
    buffers[-1].copy_to_host(host_result)
    cuda.synchronize()
    finished = perf_counter()
    return CudaRun(float(host_result[0]), allocated - start, copied - allocated,
                   calculated - copied, finished - calculated, finished - start,
                   24 + sum(buffer.nbytes for buffer in buffers))
