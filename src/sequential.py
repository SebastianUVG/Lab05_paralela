"""Integración secuencial por sumas de Riemann de punto medio."""

from math import pi

from .data import ALL_CASES, IntegrationInput


def run_sequential(inputs: IntegrationInput) -> float:
    """Aproxima la integral de 4/(1+x²) en [a,b] con n rectángulos."""

    width = (inputs.b - inputs.a) / inputs.n
    total = 0.0
    for i in range(inputs.n):
        x = inputs.a + (i + 0.5) * width
        total += 4.0 / (1.0 + x * x)
    return total * width


def main() -> None:
    for case in ALL_CASES:
        result = run_sequential(case)
        print(f"n={case.n:>9,d}  integral={result:.15f}  error_vs_pi={abs(result - pi):.3e}")


if __name__ == "__main__":
    main()
