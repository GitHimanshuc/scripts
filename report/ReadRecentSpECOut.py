#!/usr/bin/env python3

"""Print relevant lines from recent SpEC evolution segments.

This utility is strictly read-only: it only enumerates directories and opens
SpEC.out files in text read mode. Output is written to standard output.
"""

import argparse
import re
from pathlib import Path


INTERESTING = re.compile(
    r"AhC|CommonHorizon|IngoingCharFieldOnSphericalBdry|"
    r"Incoming field|Speed\[|SmoothCoordSep|Strahlkorper|"
    r"MaxIts|Termination|ERROR|FATAL",
    re.IGNORECASE,
)


def suffix_number(suffix):
    """Convert AA, AB, ..., AZ, BA, ... into increasing integers."""
    value = 0
    for character in suffix:
        value = value * 26 + ord(character) - ord("A") + 1
    return value


def find_segments(ev_path, requested_level):
    """Return SpEC.out paths grouped by numerical resolution level."""
    segments = {}

    for directory in ev_path.iterdir():
        if not directory.is_dir():
            continue

        match = re.fullmatch(r"Lev(\d+)_([A-Z]+)", directory.name)
        if not match:
            continue

        level = int(match.group(1))
        if requested_level is not None and level != requested_level:
            continue

        suffix = match.group(2)
        output = directory / "Run" / "SpEC.out"
        if output.is_file():
            segments.setdefault(level, []).append(
                (suffix_number(suffix), directory.name, output)
            )

    return segments


def print_output(filename, print_full_file):
    """Read one SpEC.out in read-only mode and print selected content."""
    found = False
    with filename.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if print_full_file or INTERESTING.search(line):
                if print_full_file:
                    print(line, end="")
                else:
                    print(f"{line_number}: {line}", end="")
                found = True

    if not found:
        print("No relevant lines found.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read Ev/LevN_XX/Run/SpEC.out from the most recent segments. "
            "The simulation directory is never modified."
        )
    )
    parser.add_argument("ev_path", type=Path, help="path to the Ev directory")
    parser.add_argument(
        "-n", "--segments", type=int, default=5, help="segments per level"
    )
    parser.add_argument("--level", type=int, help="only inspect this level")
    parser.add_argument(
        "--full", action="store_true", help="print complete SpEC.out files"
    )
    args = parser.parse_args()

    if args.segments < 1:
        parser.error("--segments must be at least 1")
    if not args.ev_path.is_dir():
        parser.error(f"not a directory: {args.ev_path}")

    segments = find_segments(args.ev_path, args.level)
    if not segments:
        parser.error(
            f"no direct LevN_XX/Run/SpEC.out files found in {args.ev_path}"
        )

    for level in sorted(segments):
        selected = sorted(segments[level])[-args.segments :]

        print(f"\n{'#' * 80}")
        print(f"LEVEL {level}: last {len(selected)} segments")
        print("#" * 80)

        for _, segment_name, filename in selected:
            print(f"\n{'=' * 80}")
            print(f"{segment_name}: {filename}")
            print("=" * 80)
            print_output(filename, args.full)


if __name__ == "__main__":
    main()
