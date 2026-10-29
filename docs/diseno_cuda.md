# Implementación CUDA y traspaso de corridas

Se conserva `IntegrationInput`, la función `4/(1+x*x)`, la regla del punto
medio y los casos de `src/data.py`. Los cálculos y sumas usan `float64`, sin
`fastmath`. La CPU multiplica por el ancho después de sumar; la GPU multiplica
cada contribución antes de reducir. Son matemáticamente equivalentes, con
posibles diferencias de redondeo.

## Entradas, grid y reducción

Se copian explícitamente dos arreglos contiguos: `[a,b]` (`float64`, 16 bytes)
y `[n]` (`int64`, 8 bytes). No se materializan los puntos de evaluación.
Los resultados parciales se reservan directamente en VRAM. Al final se copia
un solo `float64` (8 bytes) a RAM.

La grid es 1D y cada bloque tiene 256 hilos. El primer kernel usa
`ceil(n/256)` bloques; para 1 000 000 evaluaciones son **3907 bloques**.
Cada hilo calcula una contribución usando su índice global. Los hilos fuera
de rango contribuyen cero, pero participan en todas las barreras del bloque.
Cada bloque reduce sus contribuciones mediante un árbol en memoria compartida
de 256 dobles (2048 bytes), y el hilo cero escribe una suma parcial.

Los siguientes lanzamientos reducen esos parciales con el mismo árbol hasta
obtener un valor. Para 1 000 000 evaluaciones los buffers tienen 3907, 16 y 1
elementos. La memoria global reservada para los buffers y entradas suma
`24 + 8*(3907+16+1) = 31 416` bytes, sin contar recursos internos del runtime.
No hay escrituras concurrentes sobre una misma salida ni sumas finales en CPU.

256 es una potencia de dos y un múltiplo de 32, adecuada para este árbol y
sus warps. Es una configuración inicial justificada, sin afirmar que sea la
óptima para el dispositivo. El número de bloques cubre todos los índices y la
condición de límites permite tamaños no divisibles por 256.

## Transferencias y medición

`run_cuda` reserva entradas y buffers mediante `cuda.device_array`; usa
`copy_to_device` para H2D y `copy_to_host` para D2H. Se sincroniza antes de
medir y al finalizar cada fase asíncrona. El tiempo de kernel incluye todos los
lanzamientos de reducción y su coordinación desde Python, medido con
`perf_counter`; no es un tiempo obtenido mediante eventos del dispositivo.

El CSV conserva, por repetición, CPU, reservas, H2D, kernels, D2H y el total
CUDA de pared. Este total incluye reservas y coordinación, además de ambas
transferencias. No se obtiene sumando medianas. El contexto y la compilación
JIT se calientan mediante una ejecución excluida por tamaño; también se hace
una ejecución CPU de calentamiento. La generación de los parámetros, la
impresión, la validación y la liberación posterior de buffers no se cronometran.
Las reservas reflejan el comportamiento del gestor de memoria ya calentado.

Cada repetición ejecuta CPU y después GPU con exactamente los mismos
parámetros. Los tamaños se recorren en orden creciente, sin aleatorizar;
el equipo Windows usa WDDM y puede tener carga de otros procesos. El compañero
debe considerar estas condiciones al analizar variabilidad y generalizar.

## Corrección

`tests/test_correctness.py` compara CPU/GPU con `rtol=1e-11` y `atol=1e-12`.
La suma CPU de hasta un millón de términos y el árbol GPU tienen órdenes de
acumulación distintos; esta tolerancia admite pequeñas diferencias de doble
precisión sin confundirlas con el error de discretización respecto a `π`.
Se cubren los seis casos compartidos, 255, 256, 257 y 65 537 evaluaciones,
y un intervalo alternativo `[-2,3]`. Los casos CPU conocidos se prueban aparte.
Las mediciones rechazan el simulador y verifican corrección en cada fila.

## Reproducir y entregar al integrante 3

La ejecución real pasó las 6 pruebas (11 subcasos en la prueba CUDA), y las
30 corridas coincidieron con CPU dentro de tolerancia. El registro completo
está enlazado en `results/README.md`.

En esta máquina el Python global era 3.14 y no tenía Numba instalado. Se creó
`.venv` y se instalaron las bibliotecas CUDA mediante paquetes de Python.
La instalación inicial seleccionó NumPy 2.5.4, pero Numba-CUDA 0.30.4 falló
durante la compilación al buscar `numpy.row_stack`. Se resolvió fijando NumPy
2.3.5; `requirements.txt` y el archivo de dependencias completas registran
las versiones que efectivamente pasaron las pruebas. No fue necesario modificar
el controlador. Los avisos de baja ocupación se conservan en los registros.

Desde la raíz, después de instalar `requirements.txt` en `.venv`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.benchmark --trials 10
```

Cada ejecución crea una carpeta nueva `results/AAAAMMDDTHHMMSSZ/` (UTC),
con `measurements.csv` y `environment.json`. Este último incluye GPU,
controlador, Python, versiones y metodología. Una ruta ya existente se rechaza
para evitar sobrescribir evidencia. Se guardan tres tamaños y diez
repeticiones por tamaño, es decir, 30 filas medidas además del encabezado.
`speedup` es el cociente CPU/total CUDA de esa misma fila. Los resultados
numéricos y errores también se conservan, no solamente los tiempos.

Para reproducir todas las dependencias exactas, usar
`results/requirements-lock.txt`. Las corridas y su resumen de validación se
referencian en `results/README.md`. El análisis estadístico, gráficas,
interpretación de aceleración e informe quedan para el integrante 3.

Referencia de instalación:
[documentación oficial de NVIDIA Numba-CUDA](https://nvidia.github.io/numba-cuda/user/installation.html).
