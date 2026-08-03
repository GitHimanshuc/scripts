from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code_for_paper_plots/HighAccuracy1025/main_new.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("high_accuracy_main_new", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HighAccuracyMainNewTests(unittest.TestCase):
    def test_configuration_reproduces_all_non_cce_plot_names(self) -> None:
        module = _load_script_module()
        grouped_plot_count = sum(
            len(specs) for specs in module.PLOTS_BY_DIAGNOSTIC.values()
        ) * len(module.RUN_GROUPS)
        self.assertEqual(grouped_plot_count, 30)

        suffixes = {
            suffix
            for specs in module.PLOTS_BY_DIAGNOSTIC.values()
            for _, suffix in specs
        }
        self.assertIn("SphereA0_Linf_GhCe", suffixes)
        self.assertIn("VolLp(NormalizedGhCe)", suffixes)
        self.assertEqual(module.CACHE_FOLDER, Path.cwd())

    def test_plot_helper_writes_a_pdf_from_synthetic_data(self) -> None:
        module = _load_script_module()
        frame = pd.DataFrame(
            {
                "t(M)": [1205.0, 1300.0, 3999.0, 4000.0],
                "Linf(GhCe)": [1.0e-4, 8.0e-5, 2.0e-5, 1.0e-5],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            module.SAVE_FOLDER = Path(temporary_directory)
            module.save_constraint_plot(
                {"run": frame},
                {"run": "Run"},
                "Linf(GhCe)",
                "synthetic.pdf",
            )
            output = module.SAVE_FOLDER / "synthetic.pdf"
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
