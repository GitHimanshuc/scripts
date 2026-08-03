from __future__ import annotations

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code_for_paper_plots/spec_accuracy/main_new.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("spec_accuracy_main_new", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpecAccuracyMainNewTests(unittest.TestCase):
    def test_configuration_contains_all_167_non_cce_plots(self) -> None:
        module = _load_script_module()
        constraint_count = sum(
            len(specs) for specs in module.PLOTS_BY_DIAGNOSTIC.values()
        ) * len(module.RUN_GROUPS)
        power_runs, _ = module.power_runs_to_plot()
        power_count = (
            len(power_runs)
            * len(module.POWER_DOMAINS)
            * len(module.POWER_TOPOLOGIES)
            * len(module.POWER_FIELDS)
        )

        self.assertEqual(constraint_count, 36)
        self.assertEqual(power_count, 128)
        self.assertEqual(constraint_count + 3 + power_count, 167)

        output_names = {
            f"{runs_set_name}_{suffix}.pdf"
            for runs_set_name, _, _ in module.RUN_GROUPS
            for specs in module.PLOTS_BY_DIAGNOSTIC.values()
            for _, _, suffix in specs
        }
        output_names.update(
            {
                "joined_ML_5_S1_L5_SphereC28_Linf_NormalizedGhCe.pdf",
                "joined_ML_5_S1_L5_SphereC0_Linf_NormalizedGhCe.pdf",
                "joined_ML_5_S1_L5_SphereC1_Linf_NormalizedGhCe.pdf",
            }
        )
        _, power_metadata = module.power_runs_to_plot()
        output_names.update(
            f"{runs_set_name}_L{level}_PS_{field}_{domain}_{topology}.pdf"
            for runs_set_name, level, _ in power_metadata.values()
            for domain in module.POWER_DOMAINS
            for topology in module.POWER_TOPOLOGIES
            for field in module.POWER_FIELDS
        )
        self.assertEqual(len(output_names), 167)
        self.assertIn("L16_set1_SphereC22_Linf_GhCe.pdf", output_names)
        self.assertIn("L15_main_L1_PS_psi_SphereA0_0.pdf", output_names)
        self.assertIn("L16_set1_L6_PS_kappa_SphereC6_1.pdf", output_names)
        self.assertEqual(module.CACHE_FOLDER, Path.cwd())
        self.assertFalse(any(name.startswith("SKIP_") for name in vars(module)))

    def test_main_unconditionally_calls_every_plot_group(self) -> None:
        module = _load_script_module()
        normalized_linf = object()
        with tempfile.TemporaryDirectory() as temporary_directory:
            module.SAVE_FOLDER = Path(temporary_directory)
            with (
                patch.object(
                    module,
                    "make_constraint_plots",
                    return_value=normalized_linf,
                ) as constraints,
                patch.object(module, "make_individual_plots") as individual,
                patch.object(module, "make_power_plots") as power,
            ):
                module.main()

        constraints.assert_called_once_with()
        individual.assert_called_once_with(normalized_linf)
        power.assert_called_once_with()

    def test_constraint_plot_helper_writes_synthetic_pdf(self) -> None:
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
                r"$L_\infty(\mathcal{E}_{\mathrm{GH}})$",
                "constraint.pdf",
            )
            output = module.SAVE_FOLDER / "constraint.pdf"
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_power_plot_helper_writes_synthetic_pdf(self) -> None:
        module = _load_script_module()
        frame = pd.DataFrame(
            {
                "t(M)": [1210.0, 1500.0, 4000.0],
                "coef0": [1.0e-2, 1.0e-3, 1.0e-4],
                "coef1": [1.0e-4, 1.0e-5, 1.0e-6],
                "coef2": [float("nan"), float("nan"), float("nan")],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            module.SAVE_FOLDER = Path(temporary_directory)
            module.save_power_plot(
                frame,
                domain="SphereA0",
                field="psi",
                run_label="Old Level 5",
                save_name="power.pdf",
            )
            output = module.SAVE_FOLDER / "power.pdf"
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_extracted_power_pattern_loads_and_caches_synthetic_dat(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "h5_files_Lev5"
            data_path = (
                root
                / "extracted-PowerDiagnostics/SphereA0.dir/Powerpsi.dir"
                / "Bf0I1(3 modes).dat"
            )
            data_path.parent.mkdir(parents=True)
            data_path.write_text(
                '# [0] = "t"\n'
                '# [1] = "coef0"\n'
                '# [2] = "coef1"\n'
                "1210 1e-2 1e-4\n"
                "1500 1e-3 1e-5\n",
                encoding="utf-8",
            )
            module.CACHE_FOLDER = Path(temporary_directory) / "cache"
            pattern = (
                "extracted-PowerDiagnostics/SphereA0.dir/Powerpsi.dir/"
                "Bf0I1(* modes).dat"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                runs_data = module.load_diagnostic({"run": root}, pattern)

            self.assertEqual(
                list(runs_data["run"].columns),
                ["t(M)", "coef0", "coef1"],
            )
            self.assertEqual(runs_data["run"]["coef1"].tolist(), [1e-4, 1e-5])
            self.assertEqual(
                len(list(module.CACHE_FOLDER.glob("make-report-cache-*.json"))),
                1,
            )


if __name__ == "__main__":
    unittest.main()
