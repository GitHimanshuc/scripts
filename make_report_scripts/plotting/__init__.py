"""Public plotting API.

Functions accept or create explicit Matplotlib axes and never call
``plt.show()``. This keeps them usable in notebooks, batch reports, and tests.
"""

from .diagnostics import (
    plot_column_grid,
    plot_damping_times,
    plot_min_grid_spacing,
    save_column_plots,
)
from .domains import (
    BBHDomainPatchBuilder,
    BBH_domain_sym_ploy,
    get_domain_name,
    return_sorted_domain_names,
    scalar_to_color,
)
from .power import (
    plot_all_tops,
    plot_all_tops_both,
    plot_power_field_comparison,
    plot_power_heatmap,
    plot_power_spectrum,
    plot_power_topologies,
)
from .timeseries import (
    PreparedSeries,
    plot_frame_columns,
    plot_runs,
    prepare_run_series,
)

__all__ = [
    "BBHDomainPatchBuilder",
    "BBH_domain_sym_ploy",
    "PreparedSeries",
    "get_domain_name",
    "plot_column_grid",
    "plot_all_tops",
    "plot_all_tops_both",
    "plot_damping_times",
    "plot_frame_columns",
    "plot_min_grid_spacing",
    "plot_power_field_comparison",
    "plot_power_heatmap",
    "plot_power_spectrum",
    "plot_power_topologies",
    "plot_runs",
    "prepare_run_series",
    "return_sorted_domain_names",
    "save_column_plots",
    "scalar_to_color",
]
