# Actividad CUDA: sumas de Riemann CPU/GPU

Se aproxima la integral de
`f(x) = 4 / (1 + x²)` en `[0, 1]` mediante la regla del punto medio. Su valor
exacto es `π`. El repositorio cubre las tres entregas: CPU secuencial, kernel
CUDA con reducción, benchmark e [informe final](docs/informe.md).

## Ejecutar

Desde la raíz del proyecto (Windows con `.venv` o Linux/macOS con el intérprete
activo):

```bash
python -m src.sequential
python -m unittest discover -s tests -v
python -m src.benchmark --trials 10   # requiere GPU NVIDIA real
```

En PowerShell con el entorno aislado de las corridas documentadas:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.benchmark --trials 10
```

La CPU usa solo la biblioteca estándar. Los casos de `src/data.py` son
deterministas. Las corridas reales están en [`results/README.md`](results/README.md);
el diseño CUDA en [`docs/diseno_cuda.md`](docs/diseno_cuda.md); el análisis de
speedup y la discusión en [`docs/informe.md`](docs/informe.md).

## Registro previo del integrante 1 (28 de octubre de 2026)

Este registro corresponde al equipo anterior; las corridas nuevas usan otra
GPU y otro Python. Consultar `results/README.md` y `environment.json` para
la configuración medida en esta entrega.

| Elemento | Resultado |
|---|---|
| Alternativa | Equipo local, porque dispone de GPU NVIDIA; no se utilizó Colab |
| Sistema | Windows 11, compilación 26200 |
| Python | 3.12.9 |
| GPU | NVIDIA GeForce GTX 1650 Ti, 4096 MiB de VRAM |
| `nvidia-smi` | Controlador 596.36; versión CUDA anunciada por el controlador: 13.2 |
| Paquetes del entorno aislado | NumPy 2.2.3; Numba 0.67.0; Numba-CUDA 0.30.4; CUDA Toolkit 12.9.2 mediante paquetes de Python |
| `cuda.detect()` en `.venv` | Detectó un dispositivo compatible, capacidad de cómputo 7.5 |
| `cuda.is_available()` en `.venv` | `True` |

En el Python global, `cuda.is_available()` dio `False`: faltaban bibliotecas
CUDA como `nvvm.dll`, `nvrtc.dll` y `cudart.dll`. Se resolvió creando `.venv`
e instalando `numba-cuda[cu12]` y sus dependencias desde `requirements.txt`,
siguiendo la [guía de instalación de NVIDIA](https://nvidia.github.io/numba-cuda/user/installation.html).
La GPU quedó detectable en ese entorno. En ese registro inicial no se había
implementado ni ejecutado un kernel.

Para reproducir el entorno local:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "from numba import cuda; print(cuda.is_available()); cuda.detect()"
```

`nvidia-smi` informa la capacidad del controlador; la disponibilidad de las
bibliotecas CUDA se comprobó por separado en `.venv`.

## Entregas

| Hito | Contenido |
|---|---|
| H1 | `docs/flujo_secuencial.md`, `src/sequential.py`, `src/data.py`, `tests/test_sequential.py` |
| H2 | `docs/diseno_cuda.md`, `src/cuda_impl.py`, `tests/test_correctness.py` |
| H3 | `src/benchmark.py`, `results/20261029T015937Z/`, `docs/informe.md` |
