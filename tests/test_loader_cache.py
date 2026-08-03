from __future__ import annotations

import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from make_report_scripts import load_data_from_levs
from make_report_scripts.io._cache import (
    cache_path,
    make_cache_request,
    read_cache,
    write_cache,
)


def _write_dat(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    declarations = '# [0] = "t(M)"\n# [1] = "value"\n'
    values = "".join(f"{time} {value}\n" for time, value in rows)
    path.write_text(declarations + values, encoding="utf-8")


class LoadDataCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.run_root = self.root / "Ev/Lev1_A?/Run"
        self.data_path = "ConstraintNorms/Test.dat"
        self.source = self.root / "Ev/Lev1_AA/Run" / self.data_path
        self.cache_folder = self.root / "cache"
        self.runs = {"Lev1": str(self.run_root)}
        _write_dat(self.source, [(0.0, 1.0), (1.0, 2.0)])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _load(self, *, reload_cache: bool = False):
        return load_data_from_levs(
            self.runs,
            self.data_path,
            cache_folder=self.cache_folder,
            reload_cache=reload_cache,
        )

    def test_cache_is_opt_in(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            columns, data = load_data_from_levs(self.runs, self.data_path)
        self.assertEqual(caught, [])
        self.assertFalse(self.cache_folder.exists())
        self.assertEqual(list(columns), ["t(M)", "value"])
        self.assertEqual(data["Lev1"]["value"].tolist(), [1.0, 2.0])

    def test_cache_miss_creates_json_and_hit_does_not_need_source(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "is missing"):
            columns, data = self._load()
        self.assertEqual(list(columns), ["t(M)", "value"])
        assert_frame_equal(
            data["Lev1"],
            pd.DataFrame({"t(M)": [0.0, 1.0], "value": [1.0, 2.0]}),
        )

        cache_files = list(self.cache_folder.glob("make-report-cache-*.json"))
        self.assertEqual(len(cache_files), 1)
        json.loads(cache_files[0].read_text(encoding="utf-8"))

        self.source.unlink()
        with warnings.catch_warnings(record=True) as caught:
            cached_columns, cached_data = self._load()
        self.assertEqual(caught, [])
        self.assertEqual(list(cached_columns), list(columns))
        assert_frame_equal(cached_data["Lev1"], data["Lev1"])

    def test_manual_reload_controls_freshness(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, original = self._load()

        _write_dat(self.source, [(0.0, 10.0), (1.0, 20.0)])
        _, cached = self._load()
        assert_frame_equal(cached["Lev1"], original["Lev1"])

        _, refreshed = self._load(reload_cache=True)
        self.assertEqual(refreshed["Lev1"]["value"].tolist(), [10.0, 20.0])
        _, cached_again = self._load()
        assert_frame_equal(cached_again["Lev1"], refreshed["Lev1"])

    def test_deleted_or_corrupt_cache_is_rebuilt_with_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._load()
        cache_file = next(self.cache_folder.glob("make-report-cache-*.json"))

        cache_file.unlink()
        with self.assertWarnsRegex(RuntimeWarning, "is missing"):
            self._load()
        self.assertTrue(cache_file.is_file())

        cache_file.write_text("not json", encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "Could not use cache"):
            _, rebuilt = self._load()
        self.assertEqual(rebuilt["Lev1"]["value"].tolist(), [1.0, 2.0])
        json.loads(cache_file.read_text(encoding="utf-8"))

    def test_different_requests_have_different_cache_files(self) -> None:
        second_path = "ConstraintNorms/Other.dat"
        _write_dat(
            self.root / "Ev/Lev1_AA/Run" / second_path,
            [(0.0, 3.0)],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._load()
            load_data_from_levs(
                self.runs,
                second_path,
                cache_folder=self.cache_folder,
            )
        self.assertEqual(
            len(list(self.cache_folder.glob("make-report-cache-*.json"))),
            2,
        )


class CacheJsonCodecTests(unittest.TestCase):
    def test_json_round_trip_preserves_supported_dataframe_values(self) -> None:
        frame = pd.DataFrame(
            {
                "t(M)": np.array([0.0, 1.0, 2.0], dtype=np.float64),
                "integer": np.array([1, 2, 3], dtype=np.int64),
                "complex": np.array(
                    [1.0 + 2.0j, complex(math.nan, 3.0), complex(4.0, math.inf)],
                    dtype=np.complex128,
                ),
                "label": pd.Series(["first", None, "third"], dtype=object),
            }
        )
        request = make_cache_request(
            [("Lev1", ("/simulation/Ev/Lev1_A?/Run",))],
            "BondiCceR0250.h5@Beta",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = cache_path(temporary_directory, request)
            write_cache(path, request, {"Lev1": frame})
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            restored = read_cache(path, request)

        assert_frame_equal(restored["Lev1"], frame)


if __name__ == "__main__":
    unittest.main()
