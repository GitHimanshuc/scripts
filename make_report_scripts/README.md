# SpEC report tools

`make_report_scripts` contains the reusable loading, transformation, and
plotting code extracted from:

- `report/make_report_and_plots.ipynb`
- `report/power_diag.ipynb`

The exported notebook scripts remain migration references. Library modules do
not execute run-specific paths, create plots, or write files during import.

## Loading simulation data

Create one loader for a simulation's `Ev` directory:

```python
from pathlib import Path

from make_report_scripts import RunDataLoader

loader = RunDataLoader.from_ev(
    Path("/path/to/simulation/Ev"),
    "Lev5_??",
)

constraints = loader.dat("ConstraintNorms/GhCe_Linf.dat")
horizon_a = loader.horizon("AhA")
orbit = loader.orbit("OrbitDiagnostics.h5")
power = loader.power()

sphere_c0 = power.spectrum(
    subdomain="SphereC0",
    topology="Bf1S2",
    field="psi",
)
sphere_cube = power.sphere_spectrum(field="kappa")
coef10 = sphere_cube.coefficient(10)
```

`RunDataLoader` resolves segments deterministically, standardizes the time
column to `t(M)`, and defaults to keeping the later segment at overlapping
times. Choose `overlap="preserve"`, `"first"`, or `"error"` when another policy
is appropriate. For multiple run-root patterns, matches are ordered within
each root and the roots retain the order supplied by the caller.

Extracted power-diagnostic directories use a separate loader:

```python
from make_report_scripts import ExtractedPowerDiagnosticsLoader

power = ExtractedPowerDiagnosticsLoader(
    "/path/to/extracted-PowerDiagnostics"
)
flat = power.load_flat()
```

The old `read_dat_file_across_AA`, `load_data_from_levs`,
`LoadPowerDiagnostics`, and `SphereCPowerData` APIs remain available while
notebooks migrate to the explicit loaders.

### Caching bulk run data

`load_data_from_levs` can save and reuse its complete `runs_data_dict` as a
versioned JSON file. Caching is opt-in, and cache files are only written to the
folder supplied by the caller:

```python
from pathlib import Path

from make_report_scripts import load_data_from_levs

columns, runs_data = load_data_from_levs(
    runs_to_plot,
    "ConstraintNorms/GhCe_Linf.dat",
    cache_folder=Path.cwd(),
)
```

The ordered run mapping and diagnostic path determine the cache filename. A
matching cache is used without checking or requiring the original source
files. Pass `reload_cache=True` to load the sources again and atomically
replace the cache. If a cache is missing, malformed, or from an incompatible
format version, the loader warns, reloads the source data, and recreates it.
Deleting a cache is therefore also a supported way to force reconstruction.

## Plotting

Plotting functions accept explicit Matplotlib axes where useful, return the
created axes or figures, and never call `plt.show()`:

```python
import matplotlib.pyplot as plt

from make_report_scripts import plot_power_heatmap, plot_runs

fig, ax = plt.subplots()
plot_runs(
    {"Lev4": lev4, "Lev5": lev5},
    "t(M)",
    "GhCe",
    ax=ax,
    time_range=(500, 4000),
    reference="Lev5",
    absolute_difference=True,
)

fig, ax = plt.subplots()
plot_power_heatmap(
    sphere_cube,
    coefficient=10,
    ax=ax,
    time_range=(2000, 9000),
)
```

Available plotting groups are:

- `plot_runs`, `plot_frame_columns`, and `prepare_run_series` for time series
- `plot_power_heatmap`, `plot_power_spectrum`,
  `plot_power_topologies`, and `plot_power_field_comparison`
- `plot_column_grid`, `save_column_plots`, `plot_min_grid_spacing`, and
  `plot_damping_times`
- domain ordering, coloring, and BBH patch construction under
  `make_report_scripts.plotting`

The original `plot_graph_for_runs` entry points delegate to the new
implementation for compatibility.

## Notebook audit

The notebook implementations were compared definition-by-definition against
the old package. Most early reader functions were identical. The migration
also carries forward notebook additions such as:

- multiple run-root patterns in `load_data_from_levs`
- skipped `failed` paths and `math_utils::` column normalization
- extracted and joined power-diagnostic loading
- profiler multi-index loading
- segment final-time helpers

Where the notebook had a regression, the corrected behavior was retained. In
particular:

- the domain patch builder uses `self.nA`, not an undefined global `nA`
- sphere topologies are unique in new APIs rather than plotting `Bf1S2` twice
- plotting callbacks only receive `zorder` when they support it
- flat power data is aligned on `t(M)` rather than a randomly chosen row index
- format options stay attached to each legacy path pattern
- cache files are not silently written into simulation directories
