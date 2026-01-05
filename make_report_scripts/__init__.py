from .helper_functions import add_norm_constraints, num_points_per_subdomain
from .main_plot_functions import plot_graph_for_runs, plot_graph_for_runs_wrapper
from .heatmap_related_functions import return_sorted_domain_names, BBH_domain_sym_ploy, scalar_to_color

__all__ = [
    'add_norm_constraints',
    'num_points_per_subdomain',
    'plot_graph_for_runs',
    'plot_graph_for_runs_wrapper',
    'return_sorted_domain_names',
    'BBH_domain_sym_ploy',
    'scalar_to_color',
]

# ===== Sys.path Instructions for Importing from Outside This Package =====
#
# To import functions from this package in external Python files, you have several options:
#
# Option 1: Add parent directory to sys.path (recommended)
# -------------------------------------------------------
# import sys
# sys.path.append('/workspaces/spec/InputFiles')  # Add parent directory to Python path
#
# from make_report import load_data_from_levs, plot_graph_for_runs
# from make_report.helper_functions import add_norm_constraints, num_points_per_subdomain
# from make_report.main_plot_functions import plot_graph_for_runs_wrapper
#
# Alternative: Modify PYTHONPATH environment variable
# -------------------------------------------------
# Before running your Python script, set:
# export PYTHONPATH="/workspaces/spec/InputFiles:$PYTHONPATH"
#
# Then import without modifying sys.path in your code.
