"""Pruebas en GPU real; no confundir omisiones con validación CUDA."""

import math
import unittest

from numba import config, cuda

from src.cuda_impl import run_cuda
from src.data import ALL_CASES, IntegrationInput
from src.sequential import run_sequential


@unittest.skipUnless(cuda.is_available() and not config.ENABLE_CUDASIM,
                     "Se requiere una GPU CUDA real")
class CudaCorrectnessTests(unittest.TestCase):
    def test_shared_and_boundary_cases(self):
        cases = ALL_CASES + tuple(IntegrationInput(0.0, 1.0, n)
                                  for n in (255, 256, 257, 65537))
        cases += (IntegrationInput(-2.0, 3.0, 1001),)
        for case in cases:
            with self.subTest(case=case):
                actual = run_cuda(case)
                self.assertTrue(math.isclose(actual.result, run_sequential(case),
                                             rel_tol=1e-11, abs_tol=1e-12))
                self.assertGreater(actual.cuda_total_seconds, 0)


if __name__ == "__main__":
    unittest.main()
