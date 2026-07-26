"""Domain ordering, coloring, and geometry used by heatmap plots."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..heatmap_related_functions import (
    BBH_domain_sym_ploy,
    get_domain_name,
    return_sorted_domain_names,
)


class BBHDomainPatchBuilder(BBH_domain_sym_ploy):
    """PEP-8 alias for the notebook's BBH domain patch builder."""


def scalar_to_color(
    scalar_values: Mapping[str, float],
    min_max_tuple: tuple[float, float] | None = None,
    color_map: str = "viridis",
    *,
    logarithmic: bool = True,
) -> tuple[dict[str, object], ScalarMappable]:
    """Map finite domain scalars to colors and return a matching colorbar map."""

    items = [
        (key, float(value))
        for key, value in scalar_values.items()
        if np.isfinite(value) and (not logarithmic or value > 0)
    ]
    if not items:
        raise ValueError("No finite values can be mapped to colors")
    keys, raw_values = zip(*items)
    values = np.asarray(raw_values, dtype=float)
    if logarithmic:
        values = np.log10(values)

    minimum, maximum = (
        min_max_tuple
        if min_max_tuple is not None
        else (float(values.min()), float(values.max()))
    )
    if minimum == maximum:
        maximum = minimum + np.finfo(float).eps
    normalize = Normalize(vmin=minimum, vmax=maximum)
    colormap = plt.get_cmap(color_map)
    colors = {
        key: colormap(normalize(value))
        for key, value in zip(keys, values)
    }
    scalar_map = ScalarMappable(norm=normalize, cmap=colormap)
    scalar_map.set_array([])
    return colors, scalar_map


__all__ = [
    "BBHDomainPatchBuilder",
    "BBH_domain_sym_ploy",
    "get_domain_name",
    "return_sorted_domain_names",
    "scalar_to_color",
]
