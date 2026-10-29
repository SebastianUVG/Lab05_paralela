"""Entradas deterministas y compartidas para las implementaciones CPU/GPU."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class IntegrationInput:
    """Parámetros escalares para la regla del punto medio (doble precisión)."""

    a: float
    b: float
    n: int

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n <= 0:
            raise ValueError("n debe ser un entero positivo")
        if not (isfinite(self.a) and isfinite(self.b) and self.a < self.b):
            raise ValueError("se requiere un intervalo finito con a < b")


SMALL_CASES = (
    IntegrationInput(0.0, 1.0, 1),
    IntegrationInput(0.0, 1.0, 2),
    IntegrationInput(0.0, 1.0, 4),
)

PERFORMANCE_CASES = tuple(
    IntegrationInput(0.0, 1.0, n) for n in (10_000, 100_000, 1_000_000)
)

ALL_CASES = SMALL_CASES + PERFORMANCE_CASES
