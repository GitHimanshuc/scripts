import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler
from pathlib import Path

from helper_functions import (
    add_norm_constraints,
    num_points_per_subdomain,
    limit_by_col_val,
    filter_by_regex,
)
from main_plot_functions import (
    load_data_from_levs,
    plot_graph_for_runs,
    plot_graph_for_runs_wrapper,
)
from make_report_scripts.heatmap_related_functions import return_sorted_domain_names

# =========================================================================================================================================================
# ==================================================== GENERAL CONFIG ==========================================================
# =========================================================================================================================================================

save_report_base_folder = Path(
    "/work2/08330/tg876294/stampede3/reports/HighAccuracy1025/"
)
dirs_to_plot = []

# for testing change paths if we are on caltech hpc
if Path("/resnick/groups/sxs/hchaudha/scripts/make_report_scripts").exists():
    print("Running on Caltech HPC, changing paths for testing")
    save_report_base_folder = Path(
        "/resnick/groups/sxs/hchaudha/HighAccuracy1025/test_reports/"
    )
    BFI_main_folder = Path("/resnick/groups/sxs/hchaudha/HighAccuracy1025/")

    dirs_to_plot = [
        Path("/resnick/groups/sxs/hchaudha/HighAccuracy1025/HighAccuracy102501")
    ]

else:
    # My BFI main folder
    BFI_main_folder = Path("/work2/08330/tg876294/stampede3/HighAccuracy1025/")
    dirs_to_plot = [d for d in BFI_main_folder.iterdir() if d.is_dir()]

    # Nils BFI main folder
    BFI_main_folder = Path("/work2/06739/nfischer/stampede3/HighAccuracy1025/")
    dirs_to_plot += [d for d in BFI_main_folder.iterdir() if d.is_dir()]

# ==================================================== Load data and plot ==========================================================

for sim_folder in dirs_to_plot:
    print(f"Simulation folder: {sim_folder}\n")

    # %%
    # =========================================== Copy the part below for each plot type ===========================================

    print("---- Plotting dt/dT vs t(M)\n")

    save_name = "dt_over_dT.png"
    base_save_folder = save_report_base_folder / sim_folder.name
    if not base_save_folder.exists():
        base_save_folder.mkdir(parents=False, exist_ok=True)
    save_name = Path(f"{base_save_folder}/{save_name}")

    runs_to_plot = {}
    for lev in range(2, 7):
        # !!!! TODO note that we are using Ecc0 here, change if needed
        # Only add levs that exist
        if not (sim_folder / f"Ecc0/Ev/Lev{lev}_AA/Run/").exists():
            print(f"Skipping Lev{lev} as it does not exist")
            continue

        # Eventaully we need to clean up and add path support, for now convert to str
        runs_to_plot[f"{sim_folder.name}_{lev}"] = str(
            sim_folder / f"Ecc0/Ev/Lev{lev}_??/Run/"
        )

    ringdown_as_well = True
    # ringdown_as_well = False
    if ringdown_as_well:
        for key in list(runs_to_plot.keys()):
            runs_to_plot[key] = runs_to_plot[key].replace("/Ev/", "/Ev/**/")

    # psi_or_kappa = "psi"
    psi_or_kappa = "kappa"
    psi_or_kappa = "both"
    top_number = 0

    # data_file_path = "/TStepperDiag.dat"
    data_file_path = "/TimeInfo.dat"

    runs_data_dict = {}
    if psi_or_kappa == "both" and "PowerDiagnostics" in data_file_path:
        column_names_kappa, runs_data_dict_kappa = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@kappa@{top_number}"
        )
        column_names_psi, runs_data_dict_psi = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@psi@{top_number}"
        )

        for key in runs_data_dict_psi:
            runs_data_dict[key] = pd.merge(
                runs_data_dict_kappa[key],
                runs_data_dict_psi[key],
                on="t(M)",
                how="outer",
            )
        column_names = runs_data_dict[key].columns.tolist()

    elif data_file_path == "ComputeCOM":
        bhA_cols, bhA_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhA"
        )
        bhB_cols, bhB_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhB"
        )
        for key in bhA_data:
            bhA_df = bhA_data[key]
            bhB_df = bhB_data[key]
            mA = bhA_df["ChristodoulouMass"]
            mB = bhB_df["ChristodoulouMass"]
            com_x = (
                mA * bhA_df["CoordCenterInertial_0"]
                + mB * bhB_df["CoordCenterInertial_0"]
            ) / (mA + mB)
            com_y = (
                mA * bhA_df["CoordCenterInertial_1"]
                + mB * bhB_df["CoordCenterInertial_1"]
            ) / (mA + mB)
            com_z = (
                mA * bhA_df["CoordCenterInertial_2"]
                + mB * bhB_df["CoordCenterInertial_2"]
            ) / (mA + mB)
            com_df = pd.DataFrame(
                {"t(M)": bhA_df["t(M)"], "COM_X": com_x, "COM_Y": com_y, "COM_Z": com_z}
            )
            runs_data_dict[key] = com_df
            column_names = com_df.columns.tolist()
    else:
        if "PowerDiagnostics" in data_file_path:
            data_file_path = f"{data_file_path}@{psi_or_kappa}@{top_number}"
        column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    # load both psi and kappa

    # print(column_names)
    # print(runs_data_dict.keys())

    vars = set()
    domains = set()
    for run in runs_data_dict:
        for col in runs_data_dict[run]:
            vars.add(col.split(" on ")[0])
            domains.add(col.split(" on ")[1] if " on " in col else "All Domains")
    # list(sorted(vars)),list(sorted(domains))

    # #### Modify dict

    if "Constraints_Linf" in data_file_path:
        new_indices, runs_data_dict = add_norm_constraints(
            runs_data_dict,
            index_num=[1, 2, 3],
            norm=["Linf"],
            subdomains_seperately=True,
            replace_runs_data_dict=True,
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "GrDomain" in data_file_path:
        new_indices, runs_data_dict = num_points_per_subdomain(
            runs_data_dict, replace_runs_data_dict=False
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "ComputeCOM" in data_file_path:
        for key in runs_data_dict:
            df = runs_data_dict[key]
            df["COM_mag"] = np.sqrt(
                df["COM_X"] ** 2 + df["COM_Y"] ** 2 + df["COM_Z"] ** 2
            )
            runs_data_dict[key] = df
        print(runs_data_dict.keys())
        print(new_indices)

    # ### dat files plot

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = False
    modification_function = None
    append_to_title = ""
    y_axis_list = None
    x_axis = "t(M)"
    take_abs = False

    # take_abs = True

    # y_axis = 'MPI::MPwait_cum'
    # x_axis = 't'
    # y_axis = 'T [hours]'
    y_axis = "dt/dT"

    # y_axis = 'dt'

    minT = -1000

    maxT = 400000

    # if "GhCe" in y_axis:
    plot_fun = lambda x, y, label: plt.plot(x, y, label=label)

    legend_dict = {}
    for key in runs_data_dict.keys():
        legend_dict[key] = None

    # if 'Horizons.h5@' in data_file_path:
    #   append_to_title += " HorizonBH="+data_file_path.split('@')[-1]
    if "AhA" in data_file_path:
        append_to_title += " AhA"
    if "AhB" in data_file_path:
        append_to_title += " AhB"

    # with plt.style.context('default'):
    with plt.style.context("ggplot"):
        # sd = 'SphereC6'
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:20]
        # colors = plt.cm.tab10.colors
        line_styles = ["-", "--", ":", "-."]
        combined_cycler = cycler(linestyle=line_styles) * cycler(color=colors[:7])
        # combined_cycler = cycler(color=colors)*cycler(linestyle=line_styles[:3])
        plt.rcParams["axes.prop_cycle"] = combined_cycler
        #   plt.rcParams["figure.figsize"] = (15,10)
        # plt.rcParams["figure.figsize"] = (10,8)
        plt.rcParams["figure.figsize"] = (8, 6)
        # plt.rcParams["figure.figsize"] = (6,5)
        plt.rcParams["figure.autolayout"] = True
        # plt.ylim(1e-10,5e-6)
        if y_axis_list is None:
            plot_graph_for_runs(
                runs_data_dict,
                x_axis,
                y_axis,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )
        else:
            plot_graph_for_runs_wrapper(
                runs_data_dict,
                x_axis,
                y_axis_list,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )

        # plt.title("")
        #   plt.ylabel("Constraint Violations near black holes")
        #   plt.tight_layout()
        # plt.legend(loc='upper left')
        # plt.ylim(1e-14, 1e-7)
        # plt.ylim(1e-20, 1e-13)
        # save_name = f"L16_set1_ArealMass_Shift_{constant_shift_val_time}.pdf"
        # if save_name.exists():
        #     raise Exception("Change name")
        # plt.xscale('log')
        # plt.legend("")
        # plt.yscale('symlog',linthresh=1e-12)
        # plt.yscale('symlog',linthresh=1e-6)
        # plt.yscale('symlog',linthresh=1e-10)
        # plt.yscale('log')

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()
        # plt.show()

    # %%
    # =========================================== Copy the part below for each plot type ===========================================

    print("---- Plotting ProperSepHorizons vs t(M)\n")

    save_name = "ProperSepHorizons.png"
    base_save_folder = save_report_base_folder / sim_folder.name
    if not base_save_folder.exists():
        base_save_folder.mkdir(parents=False, exist_ok=True)
    save_name = Path(f"{base_save_folder}/{save_name}")

    runs_to_plot = {}
    for lev in range(2, 7):
        # !!!! TODO note that we are using Ecc0 here, change if needed
        # Only add levs that exist
        if not (sim_folder / f"Ecc0/Ev/Lev{lev}_AA/Run/").exists():
            print(f"Skipping Lev{lev} as it does not exist")
            continue
        # Eventaully we need to clean up and add path support, for now convert to str
        runs_to_plot[f"{sim_folder.name}_{lev}"] = str(
            sim_folder / f"Ecc0/Ev/Lev{lev}_??/Run/"
        )

    ringdown_as_well = True
    # ringdown_as_well = False
    if ringdown_as_well:
        for key in list(runs_to_plot.keys()):
            runs_to_plot[key] = runs_to_plot[key].replace("/Ev/", "/Ev/**/")

    # psi_or_kappa = "psi"
    psi_or_kappa = "kappa"
    psi_or_kappa = "both"
    top_number = 0

    # data_file_path = "/TStepperDiag.dat"
    data_file_path = "/ApparentHorizons/HorizonSepMeasures.dat"

    runs_data_dict = {}
    if psi_or_kappa == "both" and "PowerDiagnostics" in data_file_path:
        column_names_kappa, runs_data_dict_kappa = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@kappa@{top_number}"
        )
        column_names_psi, runs_data_dict_psi = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@psi@{top_number}"
        )

        for key in runs_data_dict_psi:
            runs_data_dict[key] = pd.merge(
                runs_data_dict_kappa[key],
                runs_data_dict_psi[key],
                on="t(M)",
                how="outer",
            )
        column_names = runs_data_dict[key].columns.tolist()

    elif data_file_path == "ComputeCOM":
        bhA_cols, bhA_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhA"
        )
        bhB_cols, bhB_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhB"
        )
        for key in bhA_data:
            bhA_df = bhA_data[key]
            bhB_df = bhB_data[key]
            mA = bhA_df["ChristodoulouMass"]
            mB = bhB_df["ChristodoulouMass"]
            com_x = (
                mA * bhA_df["CoordCenterInertial_0"]
                + mB * bhB_df["CoordCenterInertial_0"]
            ) / (mA + mB)
            com_y = (
                mA * bhA_df["CoordCenterInertial_1"]
                + mB * bhB_df["CoordCenterInertial_1"]
            ) / (mA + mB)
            com_z = (
                mA * bhA_df["CoordCenterInertial_2"]
                + mB * bhB_df["CoordCenterInertial_2"]
            ) / (mA + mB)
            com_df = pd.DataFrame(
                {"t(M)": bhA_df["t(M)"], "COM_X": com_x, "COM_Y": com_y, "COM_Z": com_z}
            )
            runs_data_dict[key] = com_df
            column_names = com_df.columns.tolist()
    else:
        if "PowerDiagnostics" in data_file_path:
            data_file_path = f"{data_file_path}@{psi_or_kappa}@{top_number}"
        column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    # load both psi and kappa

    # print(column_names)
    # print(runs_data_dict.keys())

    vars = set()
    domains = set()
    for run in runs_data_dict:
        for col in runs_data_dict[run]:
            vars.add(col.split(" on ")[0])
            domains.add(col.split(" on ")[1] if " on " in col else "All Domains")
    # list(sorted(vars)),list(sorted(domains))

    # #### Modify dict

    if "Constraints_Linf" in data_file_path:
        new_indices, runs_data_dict = add_norm_constraints(
            runs_data_dict,
            index_num=[1, 2, 3],
            norm=["Linf"],
            subdomains_seperately=True,
            replace_runs_data_dict=True,
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "GrDomain" in data_file_path:
        new_indices, runs_data_dict = num_points_per_subdomain(
            runs_data_dict, replace_runs_data_dict=False
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "ComputeCOM" in data_file_path:
        for key in runs_data_dict:
            df = runs_data_dict[key]
            df["COM_mag"] = np.sqrt(
                df["COM_X"] ** 2 + df["COM_Y"] ** 2 + df["COM_Z"] ** 2
            )
            runs_data_dict[key] = df
        print(runs_data_dict.keys())
        print(new_indices)

    # ### dat files plot

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = False
    modification_function = None
    append_to_title = ""
    y_axis_list = None
    x_axis = "t(M)"
    take_abs = False

    # take_abs = True

    # y_axis = 'MPI::MPwait_cum'
    # x_axis = 't'
    # y_axis = 'T [hours]'
    y_axis = "ProperSepHorizons"

    # y_axis = 'dt'

    minT = -1000

    maxT = 400000

    # if "GhCe" in y_axis:
    plot_fun = lambda x, y, label: plt.plot(x, y, label=label)

    legend_dict = {}
    for key in runs_data_dict.keys():
        legend_dict[key] = None

    # if 'Horizons.h5@' in data_file_path:
    #   append_to_title += " HorizonBH="+data_file_path.split('@')[-1]
    if "AhA" in data_file_path:
        append_to_title += " AhA"
    if "AhB" in data_file_path:
        append_to_title += " AhB"

    # with plt.style.context('default'):
    with plt.style.context("ggplot"):
        # sd = 'SphereC6'
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:20]
        # colors = plt.cm.tab10.colors
        line_styles = ["-", "--", ":", "-."]
        combined_cycler = cycler(linestyle=line_styles) * cycler(color=colors[:7])
        # combined_cycler = cycler(color=colors)*cycler(linestyle=line_styles[:3])
        plt.rcParams["axes.prop_cycle"] = combined_cycler
        #   plt.rcParams["figure.figsize"] = (15,10)
        # plt.rcParams["figure.figsize"] = (10,8)
        plt.rcParams["figure.figsize"] = (8, 6)
        # plt.rcParams["figure.figsize"] = (6,5)
        plt.rcParams["figure.autolayout"] = True
        # plt.ylim(1e-10,5e-6)
        if y_axis_list is None:
            plot_graph_for_runs(
                runs_data_dict,
                x_axis,
                y_axis,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )
        else:
            plot_graph_for_runs_wrapper(
                runs_data_dict,
                x_axis,
                y_axis_list,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )

        # plt.title("")
        #   plt.ylabel("Constraint Violations near black holes")
        #   plt.tight_layout()
        # plt.legend(loc='upper left')
        # plt.ylim(1e-14, 1e-7)
        # plt.ylim(1e-20, 1e-13)
        # save_name = f"L16_set1_ArealMass_Shift_{constant_shift_val_time}.pdf"
        # if save_name.exists():
        #     raise Exception("Change name")
        # plt.xscale('log')
        # plt.legend("")
        # plt.yscale('symlog',linthresh=1e-12)
        # plt.yscale('symlog',linthresh=1e-6)
        # plt.yscale('symlog',linthresh=1e-10)
        # plt.yscale('log')

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()
        # plt.show()

    # %%
    # =========================================== Copy the part below for each plot type ===========================================

    print("---- Plotting Linf(GhCe) on SphereB0 vs t(M)\n")

    data_file_path = "/ConstraintNorms/GhCe_Linf.dat"

    x_axis = "t(M)"
    y_axis = "Linf(GhCe) on SphereB0"
    take_abs = False

    save_name = f"{y_axis.replace(' ', '_')}.png"
    base_save_folder = save_report_base_folder / sim_folder.name
    if not base_save_folder.exists():
        base_save_folder.mkdir(parents=False, exist_ok=True)
    save_name = Path(f"{base_save_folder}/{save_name}")

    # =========================== PLOT CONFIGURATION ===========================

    runs_to_plot = {}
    for lev in range(2, 7):
        # !!!! TODO note that we are using Ecc0 here, change if needed
        # Only add levs that exist
        if not (sim_folder / f"Ecc0/Ev/Lev{lev}_AA/Run/").exists():
            print(f"Skipping Lev{lev} as it does not exist")
            continue
        # Eventaully we need to clean up and add path support, for now convert to str
        runs_to_plot[f"{sim_folder.name}_{lev}"] = str(
            sim_folder / f"Ecc0/Ev/Lev{lev}_??/Run/"
        )

    ringdown_as_well = True
    # ringdown_as_well = False
    if ringdown_as_well:
        for key in list(runs_to_plot.keys()):
            runs_to_plot[key] = runs_to_plot[key].replace("/Ev/", "/Ev/**/")

    # psi_or_kappa = "psi"
    psi_or_kappa = "kappa"
    psi_or_kappa = "both"
    top_number = 0

    runs_data_dict = {}
    if psi_or_kappa == "both" and "PowerDiagnostics" in data_file_path:
        column_names_kappa, runs_data_dict_kappa = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@kappa@{top_number}"
        )
        column_names_psi, runs_data_dict_psi = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@psi@{top_number}"
        )

        for key in runs_data_dict_psi:
            runs_data_dict[key] = pd.merge(
                runs_data_dict_kappa[key],
                runs_data_dict_psi[key],
                on="t(M)",
                how="outer",
            )
        column_names = runs_data_dict[key].columns.tolist()

    elif data_file_path == "ComputeCOM":
        bhA_cols, bhA_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhA"
        )
        bhB_cols, bhB_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhB"
        )
        for key in bhA_data:
            bhA_df = bhA_data[key]
            bhB_df = bhB_data[key]
            mA = bhA_df["ChristodoulouMass"]
            mB = bhB_df["ChristodoulouMass"]
            com_x = (
                mA * bhA_df["CoordCenterInertial_0"]
                + mB * bhB_df["CoordCenterInertial_0"]
            ) / (mA + mB)
            com_y = (
                mA * bhA_df["CoordCenterInertial_1"]
                + mB * bhB_df["CoordCenterInertial_1"]
            ) / (mA + mB)
            com_z = (
                mA * bhA_df["CoordCenterInertial_2"]
                + mB * bhB_df["CoordCenterInertial_2"]
            ) / (mA + mB)
            com_df = pd.DataFrame(
                {"t(M)": bhA_df["t(M)"], "COM_X": com_x, "COM_Y": com_y, "COM_Z": com_z}
            )
            runs_data_dict[key] = com_df
            column_names = com_df.columns.tolist()
    else:
        if "PowerDiagnostics" in data_file_path:
            data_file_path = f"{data_file_path}@{psi_or_kappa}@{top_number}"
        column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    # load both psi and kappa

    # print(column_names)
    # print(runs_data_dict.keys())

    vars = set()
    domains = set()
    for run in runs_data_dict:
        for col in runs_data_dict[run]:
            vars.add(col.split(" on ")[0])
            domains.add(col.split(" on ")[1] if " on " in col else "All Domains")
    # list(sorted(vars)),list(sorted(domains))

    # #### Modify dict

    if "Constraints_Linf" in data_file_path:
        new_indices, runs_data_dict = add_norm_constraints(
            runs_data_dict,
            index_num=[1, 2, 3],
            norm=["Linf"],
            subdomains_seperately=True,
            replace_runs_data_dict=True,
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "GrDomain" in data_file_path:
        new_indices, runs_data_dict = num_points_per_subdomain(
            runs_data_dict, replace_runs_data_dict=False
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "ComputeCOM" in data_file_path:
        for key in runs_data_dict:
            df = runs_data_dict[key]
            df["COM_mag"] = np.sqrt(
                df["COM_X"] ** 2 + df["COM_Y"] ** 2 + df["COM_Z"] ** 2
            )
            runs_data_dict[key] = df
        print(runs_data_dict.keys())
        print(new_indices)

    # ### dat files plot

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = False
    modification_function = None
    append_to_title = ""
    y_axis_list = None

    minT = -1000

    maxT = 400000

    # if "GhCe" in y_axis:
    plot_fun = lambda x, y, label: plt.plot(x, y, label=label)

    legend_dict = {}
    for key in runs_data_dict.keys():
        legend_dict[key] = None

    # if 'Horizons.h5@' in data_file_path:
    #   append_to_title += " HorizonBH="+data_file_path.split('@')[-1]
    if "AhA" in data_file_path:
        append_to_title += " AhA"
    if "AhB" in data_file_path:
        append_to_title += " AhB"

    # with plt.style.context('default'):
    with plt.style.context("ggplot"):
        # sd = 'SphereC6'
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:20]
        # colors = plt.cm.tab10.colors
        line_styles = ["-", "--", ":", "-."]
        combined_cycler = cycler(linestyle=line_styles) * cycler(color=colors[:7])
        # combined_cycler = cycler(color=colors)*cycler(linestyle=line_styles[:3])
        plt.rcParams["axes.prop_cycle"] = combined_cycler
        #   plt.rcParams["figure.figsize"] = (15,10)
        # plt.rcParams["figure.figsize"] = (10,8)
        plt.rcParams["figure.figsize"] = (8, 6)
        # plt.rcParams["figure.figsize"] = (6,5)
        plt.rcParams["figure.autolayout"] = True
        # plt.ylim(1e-10,5e-6)
        if y_axis_list is None:
            plot_graph_for_runs(
                runs_data_dict,
                x_axis,
                y_axis,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )
        else:
            plot_graph_for_runs_wrapper(
                runs_data_dict,
                x_axis,
                y_axis_list,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )

        # plt.title("")
        #   plt.ylabel("Constraint Violations near black holes")
        #   plt.tight_layout()
        # plt.legend(loc='upper left')
        # plt.ylim(1e-14, 1e-7)
        # plt.ylim(1e-20, 1e-13)
        # save_name = f"L16_set1_ArealMass_Shift_{constant_shift_val_time}.pdf"
        # if save_name.exists():
        #     raise Exception("Change name")
        # plt.xscale('log')
        # plt.legend("")
        # plt.yscale('symlog',linthresh=1e-12)
        # plt.yscale('symlog',linthresh=1e-6)
        # plt.yscale('symlog',linthresh=1e-10)
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()
        # plt.show()

    # %%
    # =========================================== Copy the part below for each plot type ===========================================

    print("---- Plotting Linf(GhCe) on SphereC6 vs t(M)\n")

    data_file_path = "/ConstraintNorms/GhCe_Linf.dat"

    x_axis = "t(M)"
    y_axis = "Linf(GhCe) on SphereC6"
    take_abs = False

    save_name = f"{y_axis.replace(' ', '_')}.png"
    base_save_folder = save_report_base_folder / sim_folder.name
    if not base_save_folder.exists():
        base_save_folder.mkdir(parents=False, exist_ok=True)
    save_name = Path(f"{base_save_folder}/{save_name}")

    # =========================== PLOT CONFIGURATION ===========================

    runs_to_plot = {}
    for lev in range(2, 7):
        # !!!! TODO note that we are using Ecc0 here, change if needed
        # Only add levs that exist
        if not (sim_folder / f"Ecc0/Ev/Lev{lev}_AA/Run/").exists():
            print(f"Skipping Lev{lev} as it does not exist")
            continue
        # Eventaully we need to clean up and add path support, for now convert to str
        runs_to_plot[f"{sim_folder.name}_{lev}"] = str(
            sim_folder / f"Ecc0/Ev/Lev{lev}_??/Run/"
        )

    ringdown_as_well = True
    # ringdown_as_well = False
    if ringdown_as_well:
        for key in list(runs_to_plot.keys()):
            runs_to_plot[key] = runs_to_plot[key].replace("/Ev/", "/Ev/**/")

    # psi_or_kappa = "psi"
    psi_or_kappa = "kappa"
    psi_or_kappa = "both"
    top_number = 0

    runs_data_dict = {}
    if psi_or_kappa == "both" and "PowerDiagnostics" in data_file_path:
        column_names_kappa, runs_data_dict_kappa = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@kappa@{top_number}"
        )
        column_names_psi, runs_data_dict_psi = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@psi@{top_number}"
        )

        for key in runs_data_dict_psi:
            runs_data_dict[key] = pd.merge(
                runs_data_dict_kappa[key],
                runs_data_dict_psi[key],
                on="t(M)",
                how="outer",
            )
        column_names = runs_data_dict[key].columns.tolist()

    elif data_file_path == "ComputeCOM":
        bhA_cols, bhA_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhA"
        )
        bhB_cols, bhB_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhB"
        )
        for key in bhA_data:
            bhA_df = bhA_data[key]
            bhB_df = bhB_data[key]
            mA = bhA_df["ChristodoulouMass"]
            mB = bhB_df["ChristodoulouMass"]
            com_x = (
                mA * bhA_df["CoordCenterInertial_0"]
                + mB * bhB_df["CoordCenterInertial_0"]
            ) / (mA + mB)
            com_y = (
                mA * bhA_df["CoordCenterInertial_1"]
                + mB * bhB_df["CoordCenterInertial_1"]
            ) / (mA + mB)
            com_z = (
                mA * bhA_df["CoordCenterInertial_2"]
                + mB * bhB_df["CoordCenterInertial_2"]
            ) / (mA + mB)
            com_df = pd.DataFrame(
                {"t(M)": bhA_df["t(M)"], "COM_X": com_x, "COM_Y": com_y, "COM_Z": com_z}
            )
            runs_data_dict[key] = com_df
            column_names = com_df.columns.tolist()
    else:
        if "PowerDiagnostics" in data_file_path:
            data_file_path = f"{data_file_path}@{psi_or_kappa}@{top_number}"
        column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    # load both psi and kappa

    # print(column_names)
    # print(runs_data_dict.keys())

    vars = set()
    domains = set()
    for run in runs_data_dict:
        for col in runs_data_dict[run]:
            vars.add(col.split(" on ")[0])
            domains.add(col.split(" on ")[1] if " on " in col else "All Domains")
    # list(sorted(vars)),list(sorted(domains))

    # #### Modify dict

    if "Constraints_Linf" in data_file_path:
        new_indices, runs_data_dict = add_norm_constraints(
            runs_data_dict,
            index_num=[1, 2, 3],
            norm=["Linf"],
            subdomains_seperately=True,
            replace_runs_data_dict=True,
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "GrDomain" in data_file_path:
        new_indices, runs_data_dict = num_points_per_subdomain(
            runs_data_dict, replace_runs_data_dict=False
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "ComputeCOM" in data_file_path:
        for key in runs_data_dict:
            df = runs_data_dict[key]
            df["COM_mag"] = np.sqrt(
                df["COM_X"] ** 2 + df["COM_Y"] ** 2 + df["COM_Z"] ** 2
            )
            runs_data_dict[key] = df
        print(runs_data_dict.keys())
        print(new_indices)

    # ### dat files plot

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = False
    modification_function = None
    append_to_title = ""
    y_axis_list = None

    minT = -1000

    maxT = 400000

    # if "GhCe" in y_axis:
    plot_fun = lambda x, y, label: plt.plot(x, y, label=label)

    legend_dict = {}
    for key in runs_data_dict.keys():
        legend_dict[key] = None

    # if 'Horizons.h5@' in data_file_path:
    #   append_to_title += " HorizonBH="+data_file_path.split('@')[-1]
    if "AhA" in data_file_path:
        append_to_title += " AhA"
    if "AhB" in data_file_path:
        append_to_title += " AhB"

    # with plt.style.context('default'):
    with plt.style.context("ggplot"):
        # sd = 'SphereC6'
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:20]
        # colors = plt.cm.tab10.colors
        line_styles = ["-", "--", ":", "-."]
        combined_cycler = cycler(linestyle=line_styles) * cycler(color=colors[:7])
        # combined_cycler = cycler(color=colors)*cycler(linestyle=line_styles[:3])
        plt.rcParams["axes.prop_cycle"] = combined_cycler
        #   plt.rcParams["figure.figsize"] = (15,10)
        # plt.rcParams["figure.figsize"] = (10,8)
        plt.rcParams["figure.figsize"] = (8, 6)
        # plt.rcParams["figure.figsize"] = (6,5)
        plt.rcParams["figure.autolayout"] = True
        # plt.ylim(1e-10,5e-6)
        if y_axis_list is None:
            plot_graph_for_runs(
                runs_data_dict,
                x_axis,
                y_axis,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )
        else:
            plot_graph_for_runs_wrapper(
                runs_data_dict,
                x_axis,
                y_axis_list,
                minT,
                maxT,
                legend_dict=legend_dict,
                save_path=save_path,
                moving_avg_len=moving_avg_len,
                plot_fun=plot_fun,
                diff_base=diff_base,
                plot_abs_diff=plot_abs_diff,
                constant_shift_val_time=constant_shift_val_time,
                append_to_title=append_to_title,
                modification_function=modification_function,
                take_abs=take_abs,
            )

        # plt.title("")
        #   plt.ylabel("Constraint Violations near black holes")
        #   plt.tight_layout()
        # plt.legend(loc='upper left')
        # plt.ylim(1e-14, 1e-7)
        # plt.ylim(1e-20, 1e-13)
        # save_name = f"L16_set1_ArealMass_Shift_{constant_shift_val_time}.pdf"
        # if save_name.exists():
        #     raise Exception("Change name")
        # plt.xscale('log')
        # plt.legend("")
        # plt.yscale('symlog',linthresh=1e-12)
        # plt.yscale('symlog',linthresh=1e-6)
        # plt.yscale('symlog',linthresh=1e-10)
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()
        # plt.show()

    # %%
    # =========================================== Copy the part below for each plot type ===========================================

    print("---- Plotting HeatMap Linf(GhCe) \n")

    data_file_path = "/ConstraintNorms/GhCe_Linf.dat"

    base_save_folder = save_report_base_folder / sim_folder.name
    if not base_save_folder.exists():
        base_save_folder.mkdir(parents=False, exist_ok=True)

    # =========================== PLOT CONFIGURATION ===========================

    runs_to_plot = {}
    for lev in range(2, 7):
        # !!!! TODO note that we are using Ecc0 here, change if needed
        # Only add levs that exist
        if not (sim_folder / f"Ecc0/Ev/Lev{lev}_AA/Run/").exists():
            print(f"Skipping Lev{lev} as it does not exist")
            continue
        # Eventaully we need to clean up and add path support, for now convert to str
        runs_to_plot[f"{sim_folder.name}_{lev}"] = str(
            sim_folder / f"Ecc0/Ev/Lev{lev}_??/Run/"
        )

    # ringdown_as_well = True
    # Heatmaps can't nicely handle inspiral + ringdown on the same plot
    ringdown_as_well = False
    if ringdown_as_well:
        for key in list(runs_to_plot.keys()):
            runs_to_plot[key] = runs_to_plot[key].replace("/Ev/", "/Ev/**/")

    # psi_or_kappa = "psi"
    psi_or_kappa = "kappa"
    psi_or_kappa = "both"
    top_number = 0

    runs_data_dict = {}
    if psi_or_kappa == "both" and "PowerDiagnostics" in data_file_path:
        column_names_kappa, runs_data_dict_kappa = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@kappa@{top_number}"
        )
        column_names_psi, runs_data_dict_psi = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@psi@{top_number}"
        )

        for key in runs_data_dict_psi:
            runs_data_dict[key] = pd.merge(
                runs_data_dict_kappa[key],
                runs_data_dict_psi[key],
                on="t(M)",
                how="outer",
            )
        column_names = runs_data_dict[key].columns.tolist()

    elif data_file_path == "ComputeCOM":
        bhA_cols, bhA_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhA"
        )
        bhB_cols, bhB_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhB"
        )
        for key in bhA_data:
            bhA_df = bhA_data[key]
            bhB_df = bhB_data[key]
            mA = bhA_df["ChristodoulouMass"]
            mB = bhB_df["ChristodoulouMass"]
            com_x = (
                mA * bhA_df["CoordCenterInertial_0"]
                + mB * bhB_df["CoordCenterInertial_0"]
            ) / (mA + mB)
            com_y = (
                mA * bhA_df["CoordCenterInertial_1"]
                + mB * bhB_df["CoordCenterInertial_1"]
            ) / (mA + mB)
            com_z = (
                mA * bhA_df["CoordCenterInertial_2"]
                + mB * bhB_df["CoordCenterInertial_2"]
            ) / (mA + mB)
            com_df = pd.DataFrame(
                {"t(M)": bhA_df["t(M)"], "COM_X": com_x, "COM_Y": com_y, "COM_Z": com_z}
            )
            runs_data_dict[key] = com_df
            column_names = com_df.columns.tolist()
    else:
        if "PowerDiagnostics" in data_file_path:
            data_file_path = f"{data_file_path}@{psi_or_kappa}@{top_number}"
        column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    # load both psi and kappa

    # print(column_names)
    # print(runs_data_dict.keys())

    vars = set()
    domains = set()
    for run in runs_data_dict:
        for col in runs_data_dict[run]:
            vars.add(col.split(" on ")[0])
            domains.add(col.split(" on ")[1] if " on " in col else "All Domains")
    # list(sorted(vars)),list(sorted(domains))

    # #### Modify dict

    if "Constraints_Linf" in data_file_path:
        new_indices, runs_data_dict = add_norm_constraints(
            runs_data_dict,
            index_num=[1, 2, 3],
            norm=["Linf"],
            subdomains_seperately=True,
            replace_runs_data_dict=True,
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "GrDomain" in data_file_path:
        new_indices, runs_data_dict = num_points_per_subdomain(
            runs_data_dict, replace_runs_data_dict=False
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "ComputeCOM" in data_file_path:
        for key in runs_data_dict:
            df = runs_data_dict[key]
            df["COM_mag"] = np.sqrt(
                df["COM_X"] ** 2 + df["COM_Y"] ** 2 + df["COM_Z"] ** 2
            )
            runs_data_dict[key] = df
        print(runs_data_dict.keys())
        print(new_indices)

    # ### dat files plot

    minT = -1000
    minT = 2000  # skip initial junk

    maxT = 400000

    repeated_symmetric = False
    # repeated_symmetric = True

    # Rename this eventually
    runs_data_dict_ghce = runs_data_dict

    # One plot per lev
    for key in runs_data_dict_ghce.keys():
        print(f"Plotting HeatMap for {key}")

        save_name = Path(f"{base_save_folder}/HeatMap_Linf(GhCe){key[-2:]}.png")

        data = limit_by_col_val(minT, maxT, "t(M)", runs_data_dict_ghce[key])
        if data.empty:
            print(f"No data in time range({minT}, {maxT}) for {key}, skipping")
            continue

        data = data.iloc[::1].dropna(axis=1, how="all")

        domain_col_list = filter_by_regex(
            regex=["Sphere", "Cylinder"], col_list=data.columns
        )
        domain_col_list = filter_by_regex(regex=["Linf"], col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["L2"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["\(Shift"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["\(kappaxxx"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["SphereC[4-9]"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["SphereC[2][0-9]"],col_list=data.columns)
        # domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list,exclude=True)
        domain_col_list = filter_by_regex(regex=["GhCe"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["1Con"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["2Con"],col_list=domain_col_list)
        # domain_col_list = filter_by_regex(regex=["3Con"],col_list=domain_col_list)
        visual_data = data[domain_col_list]
        # visual_data = visual_data - 1.0
        visual_data = np.log10(visual_data)

        column_names = list(visual_data.columns)
        # column_names = [i.split(" ")[-1] for i in visual_data.columns]
        column_names = return_sorted_domain_names(
            column_names, repeated_symmetric=repeated_symmetric
        )

        vmin_log, vmax_log = None, None
        # vmin_log,vmax_log = -12.417834211445228 , -8.343109330686875
        # vmin_log,vmax_log = -12.6946440611571 , -4
        # vmin_log,vmax_log = -9 , None
        if vmin_log is None:
            vmin_log = visual_data.min().min()
        if vmax_log is None:
            vmax_log = visual_data.max().max()
        print(vmin_log, ",", vmax_log)

        if repeated_symmetric:
            visual_data["Excision"] = [np.nan for i in range(len(data["t(M)"]))]

        plt.figure(figsize=(18, 10))
        imshow_plot = plt.imshow(
            visual_data[column_names],
            aspect="auto",
            cmap="RdYlGn_r",
            origin="lower",
            interpolation="none",
            vmin=vmin_log,
            vmax=vmax_log,
        )

        if repeated_symmetric:
            # Set x-ticks and labels
            plt.xticks(
                ticks=np.arange(len(column_names)),
                labels=[i.split(" ")[-1] for i in column_names],
                rotation=90,
            )
        else:
            # Set x-ticks and labels
            plt.xticks(
                ticks=np.arange(len(visual_data.columns)),
                labels=[i.split(" ")[-1] for i in column_names],
                rotation=90,
            )

        ytick_step = 1
        ytick_step = len(visual_data) // 10  # Show about 10 ticks
        plt.yticks(
            ticks=np.arange(0, len(visual_data), ytick_step),
            labels=data["t(M)"][::ytick_step].astype(int),
        )

        # Create colorbar
        colorbar = plt.colorbar(imshow_plot, label=f"{column_names[0].split(' ')[0]}")

        # Determine colorbar ticks that align with the fixed vmin and vmax
        # tick_vals = np.linspace(vmin_log, vmax_log, num=5)

        # Set these ticks on the colorbar
        # colorbar.set_ticks(tick_vals)

        # Convert ticks back to the original scale for labeling
        # colorbar.set_ticklabels([f'{10**val:.2e}' for val in tick_vals])

        plt.ylabel("t(M)")
        plt.title(f"{key}")
        plt.tight_layout()

        plt.savefig(save_name, dpi=150)
        plt.grid(False)
        plt.show()
        plt.close()
