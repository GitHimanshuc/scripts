"""Compatibility imports for the original notebook reader module.

New code should import from :mod:`make_report_scripts.io`. This module keeps
the established function names available while all implementations live in
format-specific modules.
"""

from __future__ import annotations

from .io import (
    FindMinMaxL,
    GetFiniteRadiiDataVars,
    GetFiniteRadiusExtractionList,
    GetWTDataExtracRadii,
    append_to_df,
    flatten_dict,
    hist_files_to_dataframe,
    horizon_to_pandas,
    load_horizon_data_from_levs,
    make_Bh_pandas,
    make_col_names,
    read_OrbitDiagnostics_file,
    read_WT_data,
    read_dat_file,
    read_dat_file_single_bh,
    read_dat_file_uneq_cols,
    read_finite_radius_quantaties,
    read_horizon_across_Levs,
    read_point_interpolation_file,
    read_profiler,
    read_profiler_multiindex,
)
from .io.power import PowerDiagnosticsLoader, get_top_name_from_number


def read_power_diagnostics_non_power_spectrum(
    file_path,
    dat_file_name,
    psi_or_kappa,
    top_num,
):
    """Read one non-spectrum diagnostic from a joined HDF5 file."""

    return PowerDiagnosticsLoader(
        [file_path],
        overlap="preserve",
    ).diagnostic(dat_file_name, int(top_num), psi_or_kappa)


__all__ = [
    "FindMinMaxL",
    "GetFiniteRadiiDataVars",
    "GetFiniteRadiusExtractionList",
    "GetWTDataExtracRadii",
    "append_to_df",
    "flatten_dict",
    "get_top_name_from_number",
    "hist_files_to_dataframe",
    "horizon_to_pandas",
    "load_horizon_data_from_levs",
    "make_Bh_pandas",
    "make_col_names",
    "read_OrbitDiagnostics_file",
    "read_WT_data",
    "read_dat_file",
    "read_dat_file_single_bh",
    "read_dat_file_uneq_cols",
    "read_finite_radius_quantaties",
    "read_horizon_across_Levs",
    "read_point_interpolation_file",
    "read_power_diagnostics_non_power_spectrum",
    "read_profiler",
    "read_profiler_multiindex",
]
