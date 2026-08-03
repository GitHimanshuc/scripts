from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from make_report_scripts.io import read_dat_file


class DatHeaderReaderTests(unittest.TestCase):
    def test_unquoted_column_names_retain_spaces_and_remain_unique(self) -> None:
        contents = """\
# [0] = t(M)
# [1] = Linf(GhCe)
# [2] = Linf(GhCe) on SphereA0
# [3] = Linf(GhCe) on SphereA1
# [4] = Linf(GhCe) on SphereA2
0 1 2 3 4
1 5 6 7 8
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "GhCe_Linf.dat"
            path.write_text(contents, encoding="utf-8")
            frame = read_dat_file(path)

        self.assertEqual(
            list(frame.columns),
            [
                "t(M)",
                "Linf(GhCe)",
                "Linf(GhCe) on SphereA0",
                "Linf(GhCe) on SphereA1",
                "Linf(GhCe) on SphereA2",
            ],
        )
        self.assertEqual(frame["Linf(GhCe) on SphereA2"].tolist(), [4, 8])

    def test_quoted_names_and_inline_comments_still_parse(self) -> None:
        contents = """\
# [0] = "time after step"
# [1] = 'quoted value name'
# [2] = unquoted value name   # explanatory comment
0 1 2
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "Diagnostic.dat"
            path.write_text(contents, encoding="utf-8")
            frame = read_dat_file(path)

        self.assertEqual(
            list(frame.columns),
            ["time after step", "quoted value name", "unquoted value name"],
        )


if __name__ == "__main__":
    unittest.main()
