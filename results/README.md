# Corridas reales para análisis del integrante 3

Se ejecutaron el **28 de octubre de 2026 a las 19:59, hora de Guatemala**.
El nombre de la carpeta usa UTC, por eso indica el día 29.

- [CSV de las 30 corridas](20261029T015937Z/measurements.csv): diez por cada
  `n = 10 000, 100 000, 1 000 000`, todas con `a=0`, `b=1`.
- [Entorno y metodología](20261029T015937Z/environment.json): GPU real,
  versiones, controlador, hashes SHA-256 del código y exclusiones del tiempo.
- [Salida de las pruebas](validation.txt): 6 pruebas correctas, incluyendo
  una prueba CUDA con 11 subcasos, sin omisiones.
- [Registro del benchmark](benchmark.log).
- [Verificación del CSV](data_validation.json): 30 filas correctas,
  diez por tamaño, tiempos positivos y suma de fases consistente con el total.
- [Dependencias exactas](requirements-lock.txt).

Equipo utilizado: Windows 11, Python 3.14.0, NVIDIA GeForce RTX 3050 6GB
Laptop GPU, capacidad de cómputo 8.6, controlador 592.82. El controlador
anuncia CUDA 13.1; las bibliotecas instaladas para la ejecución son CUDA 12.9.
No se usó Colab ni el simulador de CUDA.

Cada fila contiene los tiempos CPU, reserva de memoria, H2D, kernels y D2H,
total CUDA, memoria global reservada, resultados CPU/GPU, error absoluto,
validación y cociente CPU/total CUDA. Todos los tiempos se expresan en segundos.
El mayor error absoluto CPU/GPU observado fue `1.1235457009206584e-13`,
dentro de `rtol=1e-11`, `atol=1e-12` en todas las filas.

El total CUDA incluye las reservas además de ambas transferencias y todos los
kernels de reducción. Para analizar aceleración usar `cuda_total_seconds`,
y para cada tamaño resumir sus diez repeticiones. No sustituir ese total por
el tiempo de kernel ni por una suma de medianas de fases.

Numba emitió avisos de baja ocupación para grids pequeños, incluidos los
últimos pasos de reducción. Los avisos se conservan en los registros; no fueron
errores de ejecución. Esta configuración no se ajustó mediante una búsqueda
de rendimiento. Los efectos de WDDM, carga del equipo y frecuencia de la GPU
deben considerarse al interpretar las mediciones.

Resumen robusto (medianas por tamaño):
[`summary_medians.csv`](summary_medians.csv). El informe final con tabla
comparativa, cálculo de speedup y discusión está en
[`docs/informe.md`](../docs/informe.md).

Para repetir desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r results/requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.benchmark --trials 10
```

Usar Python 3.14 en Windows x64 para reproducir este entorno exacto.
Cada benchmark crea una carpeta nueva y conserva las corridas anteriores.
Diseño CUDA: [`docs/diseno_cuda.md`](../docs/diseno_cuda.md).
