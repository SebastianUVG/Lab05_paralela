# Actividad CUDA: sumas de Riemann CPU/GPU

Se aproxima la integral de
`f(x) = 4 / (1 + x²)` en `[0, 1]` mediante la regla del punto medio. Su valor
exacto es `π`. Se incluyen las implementaciones CPU y CUDA, pruebas y un
recolector de corridas. El análisis de los datos y el informe final quedan
para el integrante 3.

## Ejecutar GPU y consultar corridas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.benchmark --trials 10
```

Las corridas reales y la configuración de esta máquina se referencian en
[`results/README.md`](results/README.md). Cada ejecución guarda un CSV nuevo,
con diez repeticiones por tamaño, y metadatos del entorno. La metodología,
el diseño de kernels y las instrucciones de traspaso están en
[`docs/diseno_cuda.md`](docs/diseno_cuda.md).

## Ejecutar

Desde esta carpeta, con Python 3.10 o posterior:

```powershell
python -m src.sequential
python -m unittest discover -s tests -v
```

La implementación CPU usa solamente la biblioteca estándar. El entorno aislado
`.venv` incluye las dependencias de `requirements.txt` para ejecutar CUDA.
Los casos de entrada se definen en `src/data.py` y son
deterministas: no se generan muestras aleatorias ni se materializan arreglos.

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

## Entrega H1

- Algoritmo, entradas, tamaños y memoria: `docs/flujo_secuencial.md`.
- Implementación CPU: `src/sequential.py`.
- Casos compartidos para CPU/GPU: `src/data.py`.
- Pruebas CPU: `tests/test_sequential.py`.

El integrante 2 debe importar `IntegrationInput` y `PERFORMANCE_CASES` desde
`src.data`, respetar la misma función, intervalo, precisión `float64` y regla
del punto medio, y comparar con `run_sequential(case)`.
