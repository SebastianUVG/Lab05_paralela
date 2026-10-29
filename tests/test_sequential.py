"""Casos de resultado conocido calculados sin depender de CUDA."""

import math
import unittest

from src.data import ALL_CASES, IntegrationInput, PERFORMANCE_CASES, SMALL_CASES
from src.sequential import run_sequential


class SequentialTests(unittest.TestCase):
    def test_one_rectangle(self) -> None:
        self.assertAlmostEqual(run_sequential(SMALL_CASES[0]), 3.2, places=14)

    def test_two_rectangles_manual_sum(self) -> None:
        # 0.5 * (4/(1+0.25**2) + 4/(1+0.75**2))
        expected = 0.5 * (64.0 / 17.0 + 64.0 / 25.0)
        self.assertAlmostEqual(run_sequential(SMALL_CASES[1]), expected, places=14)

    def test_convergence_to_pi(self) -> None:
        errors = [abs(run_sequential(case) - math.pi) for case in SMALL_CASES]
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[1], errors[0])

    def test_shared_inputs(self) -> None:
        self.assertEqual(len(ALL_CASES), 6)
        self.assertEqual([case.n for case in PERFORMANCE_CASES], [10_000, 100_000, 1_000_000])
        self.assertTrue(all((case.a, case.b) == (0.0, 1.0) for case in ALL_CASES))

    def test_invalid_inputs(self) -> None:
        for n in (0, -1, 1.5, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                IntegrationInput(0.0, 1.0, n)
        with self.assertRaises(ValueError):
            IntegrationInput(1.0, 0.0, 2)
        with self.assertRaises(ValueError):
            IntegrationInput(float("nan"), 1.0, 2)


if __name__ == "__main__":
    unittest.main()
