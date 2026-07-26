#!/usr/bin/env python3

"""Report the ExpCheb half-power on SphereC S2 modes.

The input is a FilterDiagnostics.h5 file written by
SpectralFilterControllers::ExpCheb.  Output is CSV, with one column for each
SphereC shell:

    Event,Time,SphereC0,SphereC1,...

A positive value means that ExpCheb was active on that shell's S2 modes.  Zero
means that no ExpCheb component was active; it does not mean that the fixed
Heaviside (KillTop) filter was inactive.  A blank field means that the shell
has no sample for that diagnostic event.

Repeated times are expected during domain initialization.  The Event column
distinguishes successive filter changes at the same simulation time.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError:
    sys.exit(
        "ReportExpFilteringForS2.py requires h5py. Activate a Python "
        "environment containing h5py and run the command again."
    )


GROUP_NAME = "SubdomainFilters.dir/ExpChebCoef.dir"
SPHERE_NAME = re.compile(r"^SphereC([0-9]+)[.]dat$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Print the ExpCheb half-power on the S2 modes of every SphereC "
            "shell in FilterDiagnostics.h5."
        )
    )
    parser.add_argument("file", type=Path, help="Path to FilterDiagnostics.h5")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Print only events at which at least one SphereC S2 value is positive.",
    )
    return parser.parse_args()


def decode_legend(dataset):
    if "Legend" not in dataset.attrs:
        raise ValueError(f"{dataset.name} has no Legend attribute")
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in dataset.attrs["Legend"]
    ]


def format_number(value):
    if value == int(value):
        return str(int(value))
    return f"{value:.16g}"


def read_sphere_series(filename):
    series = {}
    with h5py.File(filename, "r") as h5file:
        if GROUP_NAME not in h5file:
            raise ValueError(f"{filename} has no /{GROUP_NAME} group")

        group = h5file[GROUP_NAME]
        sphere_datasets = []
        for dataset_name in group:
            match = SPHERE_NAME.fullmatch(dataset_name)
            if match:
                sphere_datasets.append(
                    (int(match.group(1)), dataset_name, group[dataset_name])
                )
        sphere_datasets.sort()

        if not sphere_datasets:
            raise ValueError(
                f"{filename} contains no /{GROUP_NAME}/SphereC<number>.dat datasets"
            )

        for _, dataset_name, dataset in sphere_datasets:
            legend = decode_legend(dataset)
            try:
                time_column = legend.index("Time")
            except ValueError as error:
                raise ValueError(f"{dataset.name} has no Time column") from error

            s2_columns = [
                index for index, label in enumerate(legend) if label.endswith("S2")
            ]
            if len(s2_columns) != 1:
                raise ValueError(
                    f"{dataset.name} should have exactly one S2 column; "
                    f"its Legend is {legend}"
                )
            s2_column = s2_columns[0]

            occurrence_at_time = defaultdict(int)
            values = {}
            for row in dataset:
                time = float(row[time_column])
                event_key = (time, occurrence_at_time[time])
                occurrence_at_time[time] += 1
                values[event_key] = float(row[s2_column])

            series[dataset_name.removesuffix(".dat")] = values

    return series


def main():
    args = parse_args()
    try:
        series = read_sphere_series(args.file)
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")

    event_keys = sorted(
        {event_key for values in series.values() for event_key in values}
    )
    sphere_names = list(series)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["Event", "Time", *sphere_names])

    active_events = 0
    for event, event_key in enumerate(event_keys):
        values = [series[name].get(event_key) for name in sphere_names]
        is_active = any(value is not None and value > 0 for value in values)
        if is_active:
            active_events += 1
        if args.active_only and not is_active:
            continue

        time, _ = event_key
        writer.writerow(
            [
                event,
                format_number(time),
                *[
                    "" if value is None else format_number(value)
                    for value in values
                ],
            ]
        )

    sys.stdout.flush()
    if active_events:
        print(
            f"Found active SphereC S2 ExpCheb filtering at {active_events} "
            "diagnostic event(s).",
            file=sys.stderr,
        )
    else:
        print(
            "No active SphereC S2 ExpCheb filtering was found.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
