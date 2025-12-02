# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler
from pathlib import Path

from helper_functions import add_norm_constraints, num_points_per_subdomain
from main_plot_functions import load_data_from_levs, plot_graph_for_runs, plot_graph_for_runs_wrapper

# =========================================================================================================================================================
# =========================================================================================================================================================
# =========================================================================================================================================================

runs_to_plot = {}
save_report_base_folder = Path("/work2/08330/tg876294/stampede3/reports/HighAccuracy1025/HighAccuracy102501/")
if not save_report_base_folder.exists():
    save_report_base_folder.mkdir(parents=False, exist_ok=True)

# runs_to_plot['asd']="/workspaces/spec/Tests/BlackBoxTests/GeneralizedHarmonicExamples/BBHLong/Save/Lev0_AA/"

for lev in range(2, 7):
    runs_to_plot[f"HighAccuracy102501_{lev}"] = (
        f"/work2/08330/tg876294/stampede3/HighAccuracy1025/HighAccuracy102501/Ecc0/Ev/Lev{lev}_??/Run/"
    )

ringdown_only = True
ringdown_only = False
if ringdown_only:
    for key in list(runs_to_plot.keys()):
        runs_to_plot[key] = runs_to_plot[key].replace("/Ev/", "/Ev/**/")

# data_file_path = "ConstraintNorms/GhCe.dat"
# data_file_path = "ConstraintNorms/GhCeExt.dat"
# data_file_path = "ConstraintNorms/GhCeExt_L2.dat"
# data_file_path = "ConstraintNorms/GhCeExt_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_L2.dat"
data_file_path = "ConstraintNorms/GhCe_Linf.dat"
# data_file_path = "ConstraintNorms/Linf.dat"
# data_file_path = "ConstraintNorms/Constraints_Linf.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path = "ConstraintNorms/GhCe_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_VolL2.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Norms.dat"
# data_file_path = "CharSpeedNorms/CharSpeeds_Min_SliceLFF.SphereA0.dat"
# data_file_path = "MinimumGridSpacing.dat"
# data_file_path = "GrAdjustMaxTstepToDampingTimes.dat"
# data_file_path = "GrAdjustSubChunksToDampingTimes.dat"
# data_file_path = "DiagAhSpeedA.dat"
# data_file_path = "ApparentHorizons/AhA.dat"
# data_file_path = "ApparentHorizons/AhB.dat"
# data_file_path = "ApparentHorizons/MinCharSpeedAhA.dat"
# data_file_path = "ApparentHorizons/RescaledRadAhA.dat"
# data_file_path = "ApparentHorizons/AhACoefs.dat"
# data_file_path = "ApparentHorizons/AhBCoefs.dat"
# data_file_path = "ApparentHorizons/Trajectory_AhB.dat"
# data_file_path = "ApparentHorizons/Trajectory_AhA.dat"
# data_file_path = "ApparentHorizons/HorizonSepMeasures.dat"
# data_file_path = "ConstraintNorms/Metric_norms.dat"
# data_file_path = "ConstraintNorms/Metric_L2.dat"

# data_file_path = "PowerDiagnostics.h5@ConvergenceFactor.dat"
# data_file_path = "PowerDiagnostics.h5@HighestThirdConvergenceFactor.dat"
# data_file_path = "PowerDiagnostics.h5@NumberOfFilteredModes.dat"
# data_file_path = "PowerDiagnostics.h5@NumberOfModes.dat"
# data_file_path = "PowerDiagnostics.h5@NumberOfNonFilteredNonZeroModes.dat"
# data_file_path = "PowerDiagnostics.h5@NumberOfPiledUpModes.dat"
# data_file_path = "PowerDiagnostics.h5@PowerInFilteredModes.dat"
# data_file_path = "PowerDiagnostics.h5@PowerInHighestUnfilteredModes.dat"
# data_file_path = "PowerDiagnostics.h5@PredictedTruncationErrorForOneLessMode.dat"
# data_file_path = "PowerDiagnostics.h5@RawConvergenceFactor.dat"
# data_file_path = "PowerDiagnostics.h5@SpectrumIsDegenerate.dat"
# data_file_path = "PowerDiagnostics.h5@TruncationError.dat"

# psi_or_kappa = "psi"
psi_or_kappa = "kappa"
psi_or_kappa = "both"
top_number = 0

# data_file_path = "ApparentHorizons/Horizons.h5@AhA"
# data_file_path = "ApparentHorizons/Horizons.h5@AhB"
# data_file_path = "ComputeCOM"
# data_file_path = "TStepperDiag.dat"
data_file_path = "TimeInfo.dat"
# data_file_path = "Hist-FuncSkewAngle.txt"
# data_file_path = "Hist-FuncCutX.txt"
# data_file_path = "Hist-FuncExpansionFactor.txt"
# data_file_path = "Hist-FuncLambdaFactorA0.txt"
# data_file_path = "Hist-FuncLambdaFactorA.txt"
# data_file_path = "Hist-FuncLambdaFactorB0.txt"
# data_file_path = "Hist-FuncLambdaFactorB.txt"
# data_file_path = "Hist-FuncQuatRotMatrix.txt"
# data_file_path = "Hist-FuncSkewAngle.txt"
# data_file_path = "Hist-FuncSmoothCoordSep.txt"
# data_file_path = "Hist-FuncSmoothMinDeltaRNoLam00AhA.txt"
# data_file_path = "Hist-FuncSmoothMinDeltaRNoLam00AhB.txt"
# data_file_path = "Hist-FuncSmoothRAhA.txt"
# data_file_path = "Hist-FuncSmoothRAhB.txt"
# data_file_path = "Hist-FuncTrans.txt"
# data_file_path = "Hist-GrDomain.txt"
# data_file_path = "Profiler.h5"
# data_file_path = "OrbitDiagnostics.h5"


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
            runs_data_dict_kappa[key], runs_data_dict_psi[key], on="t(M)", how="outer"
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
            mA * bhA_df["CoordCenterInertial_0"] + mB * bhB_df["CoordCenterInertial_0"]
        ) / (mA + mB)
        com_y = (
            mA * bhA_df["CoordCenterInertial_1"] + mB * bhB_df["CoordCenterInertial_1"]
        ) / (mA + mB)
        com_z = (
            mA * bhA_df["CoordCenterInertial_2"] + mB * bhB_df["CoordCenterInertial_2"]
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

print(column_names)
print(runs_data_dict.keys())

# %%
vars = set()
domains = set()
for run in runs_data_dict:
    for col in runs_data_dict[run]:
        vars.add(col.split(" on ")[0])
        domains.add(col.split(" on ")[1] if " on " in col else "All Domains")
# list(sorted(vars)),list(sorted(domains))

# %% [markdown]
# #### Modify dict

# %%
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
        df["COM_mag"] = np.sqrt(df["COM_X"] ** 2 + df["COM_Y"] ** 2 + df["COM_Z"] ** 2)
        runs_data_dict[key] = df
    print(runs_data_dict.keys())
    print(new_indices)


# %% [markdown]
# ### dat files plot

# %%
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

take_abs = True

# diff_base = '17_set1_9_18_L3_correct'
# diff_base = '36_segs_L6'
# diff_base = '6_set1_L6s6'
# diff_base = '34_master_L16_6'
# diff_base = 'q1_0_0_8_0_0_8_MaxDL'
# diff_base = 'L3AH500000'
# add_max_and_min_val(runs_data_dict)
# y_axis = 'max_val'
# y_axis = 'min_val'

# constant_shift_val_time = 1200
# constant_shift_val_time = 11790
# constant_shift_val_time = 1150
# constant_shift_val_time = 2000
# constant_shift_val_time = 5900
# constant_shift_val_time = 6000
# constant_shift_val_time = 7000
# if constant_shift_val_time is not None:
#     take_abs = True

# sd = 'SphereD0'
# sd = 'SphereD5'
# sd = 'SphereE0'
# sd = 'SphereE10'
# sd = 'SphereB0'
# sd = 'SphereB4'

# sd = 'SphereA0'
# sd = 'SphereB0'
# sd = 'SphereC4'
sd = "SphereC0"
# sd = 'SphereC4'
# sd = 'SphereC10'
# sd = 'SphereC15'
# sd = 'SphereC20'
# sd = 'SphereC25'
# sd = 'SphereC28'
# sd = 'SphereC45'
# sd = 'CylinderEB0.0.0'
# sd = 'CylinderEB1.0.0'
# sd = 'CylinderCB0.0.0'
# sd = 'CylinderCB1.0.0'
# sd = 'FilledCylinderEB0'
# sd = 'FilledCylinderEB1'
# sd = 'FilledCylinderCB0'
# sd = 'FilledCylinderCB1'
# sd = 'FilledCylinderMB0'
# sd = 'FilledCylinderMB1'
# sd = 'CylinderSMB0.0'
# sd = 'CylinderSMB1.0'
# sd = 'CylinderA1EA0.0'

y_axis = f"Linf(GhCe) on {sd}"

# y_axis_list = [f'Linf(GhCe) on SphereC5',f'Linf(GhCe) on SphereC4',f'Linf(GhCe) on SphereC3' ]
# y_axis = f'Linf(1Con) on {sd}'
# y_axis = f'Linf(2Con) on {sd}'
# y_axis = f'Linf(3Con) on {sd}'
# y_axis_list = [f'Linf(1Con) on {sd}',f'Linf(2Con) on {sd}',f'Linf(3Con) on {sd}']
# y_axis = f'Linf(sqrt(1Con^2)) on {sd}'
# y_axis = f'Linf(sqrt(2Con^2)) on {sd}'
# y_axis = f'Linf(sqrt(3Con^2)) on {sd}'
# y_axis_list = [f'Linf(sqrt(1Con^2)) on {sd}',f'Linf(sqrt(2Con^2)) on {sd}',f'Linf(sqrt(3Con^2)) on {sd}']
# y_axis_list = [i for i in column_names if f'Linf(GhCe)' in i and 'SphereC' in i]
# y_axis_list = [f'L2(1Con) on {sd}',f'L2(2Con) on {sd}',f'L2(3Con) on {sd}'] + y_axis_list
# y_axis = f'Linf(sqrt(kappa^2)) on {sd}'
# y_axis = f'L2(kappatxx) on {sd}'
# y_axis = f'L2(sqrt(psi^2)) on {sd}'
# y_axis = f'L2(sqrt(kappa^2)) on {sd}'
# y_axis_list = [f'Linf({i}Con) on {sd}' for i in [1,2,3]]
# y_axis = f'Linf(GhCe) on {sd}'
# y_axis = f'Linf(NormalizedGhCe) on {sd}'
# y_axis = f'MinimumGridSpacing[{sd}]'
# y_axis = f'NumPoints in {sd}'
# y_axis = f'{psi_or_kappa}_{data_file_path.split("@")[1][:-4]}_{get_top_name_from_number(top_number,sd)} on {sd}'
# y_axis = f'psi_{data_file_path.split("@")[1][:-4]}_{get_top_name_from_number(top_number,sd)} on {sd}'
# y_axis = f'kappa_{data_file_path.split("@")[1][:-4]}_{get_top_name_from_number(top_number,sd)} on {sd}'
# if psi_or_kappa == "both":
#     y_axis_list = [f'kappa_{data_file_path.split("@")[1][:-4]}_{get_top_name_from_number(top_number,sd)} on {sd}',f'psi_TruncationError_{get_top_name_from_number(top_number,sd)} on {sd}']
# y_axis = 'Linf(sqrt(3Con^2)) on SphereD0'
# y_axis = 'Linf(sqrt(3Con^2)) on SphereE5'

fund_var = "kappa"
fund_var = "psi"
# var = 'PowerInHighestUnfilteredModes'
var = "TruncationError"
# y_axis = f'{fund_var}_{var}_{get_top_name_from_number(top_number,sd)} on {sd}'
# y_axis_list = [f'psi_{var}_{get_top_name_from_number(top_number,sd)} on {sd}',f'kappa_{var}_{get_top_name_from_number(top_number,sd)} on {sd}']

# y_axis = f'{sd}_R'
# y_axis = f'{sd}_L'
# y_axis_list = [f'{sd}_R',f'{sd}_L']
# y_axis_list = [f'SphereC5_L',f'SphereC4_L']
# y_axis = f'{sd}_M'

# y_axis_list = [f'Linf(GhCe) on SphereC{n}' for n in range(26)]

# y_axis = 'Linf(GhCe)'
# y_axis = 'Linf(NormalizedGhCe)'

# y_axis = 'phi'

# y_axis = 'Linf(3Conzzz) on CylinderSMB0.0'

# y_axis = 'MPI::MPwait_cum'
# x_axis = 't'
# y_axis = 'T [hours]'
y_axis = "dt/dT"

# y_axis = 'FilledCylinderMB0_L'
# y_axis = 'MinimumGridSpacing[SphereA0]'
# y_axis = 'MinimumGridSpacing[CylinderSMB0.0]'
# y_axis = 'CoordSepHorizons'

# y_axis = 'ArealMass'
# y_axis = 'ChristodoulouMass'
# y_axis = 'CoordCenterInertial_0'
# y_axis = 'CoordCenterInertial_2'
# y_axis = 'CoordSpinChiMagInertial'
# y_axis = 'DimensionfulInertialCoordSpinMag'
# y_axis = 'DimensionfulInertialSpinMag'
# y_axis = 'chiMagInertial'
# y_axis = 'min(r)'
# y_axis = 'max(r)'
# y_axis = 'NumIterations'
# y_axis = 'L_surface'
# y_axis = 'L_mesh'
# y_axis = 'L_max'
# y_axis = 'Residual'
# y_axis = 'Shape_TruncationError'
# y_axis = 'InertialCenter_x'

# point = "(0,0,0)"
# y_axis = f"{data_file_path.split('/')[-1][4:-4]}_{point}"

# y_axis = 'courant factor'
# y_axis = 'error/1e-08'
# y_axis = 'NfailedSteps'
# y_axis = 'NumRhsEvaluations in this segment'
# y_axis = 'dt'
# y_axis = 'dt desired by control'
# y_axis_list = ['dt','dt desired by control']
# y_axis = 'FilledCylinderMA1_L'
# y_axis = 'NumIterations'
# y_axis = 'ProperSepHorizons'
# y_axis = 'dt'

# x_axis = 'COM_X'
# y_axis = 'COM_X'
# y_axis = 'COM_Y'
# y_axis = 'COM_mag'

minT = -1000
# minT = 0
minT = 85
# minT = 750
# minT = constant_shift_val_time-50
# minT = 1500
# minT = 4100
# minT = 6001
# minT = 6100
# minT = 8000
# minT = 8700
# minT = 11000

maxT = 400000
# maxT = minT+5
# maxT = 700
# maxT = minT + 250
# maxT = 2777
# maxT = 4000
# maxT = 6300
# maxT = 9500
# maxT = 11500
# moving_avg_len = 10
# moving_avg_len = 50
# moving_avg_len = 1000
# maxT = 47100
# maxT = 46600


# y_axis_list = [f"SphereC{i}_R" for i in range(30)]
# y_axis_list = ["SphereC0_L","SphereC1_L","SphereC2_L","SphereC4_L","SphereC8_L","SphereC16_L","SphereC29_L"]
# y_axis_list = ["SphereC4_L","SphereC16_L","SphereC29_L"]
# y_axis_list = [f'Linf(GhCe) on SphereA{i}' for i in [0]]
# y_axis_list = ["SphereC0_R",'CylinderSMA0.0_R','FilledCylinderMA0_R','SphereA0_R']
# y_axis_list = ['SphereA0_L','SphereA1_L','SphereA2_L','SphereA3_L','SphereA4_L']
# y_axis_list = ['Linf(GhCe) on CylinderSMA0.0','Linf(GhCe) on FilledCylinderMA0','Linf(GhCe) on SphereA0']
# y_axis_list = [
#   'Linf(NormalizedGhCe) on SphereA0',
#   'Linf(NormalizedGhCe) on SphereA1',
#   'Linf(NormalizedGhCe) on SphereA2',
#   'Linf(NormalizedGhCe) on SphereA3',
#   'Linf(NormalizedGhCe) on SphereA4',
#   'Linf(NormalizedGhCe) on CylinderSMA0.0',
#   'Linf(NormalizedGhCe) on FilledCylinderMA0',
#   'Linf(NormalizedGhCe) on SphereC0',
#   'Linf(NormalizedGhCe) on SphereC1',
#   'Linf(NormalizedGhCe) on SphereC2',
#   'Linf(NormalizedGhCe) on SphereC4',
#   'Linf(NormalizedGhCe) on SphereC8',
#   'Linf(NormalizedGhCe) on SphereC12',
#   'Linf(NormalizedGhCe) on SphereC16',
#   'Linf(NormalizedGhCe) on SphereC20',
#   'Linf(NormalizedGhCe) on SphereC24',
#   'Linf(NormalizedGhCe) on SphereC28',
#   ]
# y_axis_list = [f'Linf(1Con{v}) on SphereC0' for v in ['t','x','y','z']]
# y_axis_list = [f'Linf(sqrt(kappaErr^2)) on SphereC{i}' for i in range(0,12)]
# y_axis_list = [f'Linf(NormalizedGhCe) on SphereC{i}' for i in range(5,45,10)]
# y_axis_list = [f'Linf(NormalizedGhCe) on SphereA{i}' for i in range(0,5)]
# y_axis_list = [f'Linf(GhCe) on SphereA{i}' for i in range(0,5)]
# y_axis_list = [f'Linf(GhCe) on SphereC{i}' for i in range(0,45,2)]
# y_axis_list = [f'Linf(GhCe) on SphereC{i}' for i in range(0,45,3)]
# y_axis_list = ['MinimumGridSpacing[CylinderSMA0.0]','MinimumGridSpacing[FilledCylinderMA0]','MinimumGridSpacing[SphereA0]']
# y_axis_list = [i for i in column_names if ('SphereA' in i)]
# y_axis_inc_list = [f"SphereC{i}$" for i in range(0,45,5)]
# y_axis_list = []
# for col in column_names:
#   for i in y_axis_inc_list :
#     if re.search(i,col):
#       y_axis_list.append(col)
# print(y_axis_list)
# y_axis_list = column_names

# if "GhCe" in y_axis:
plot_fun = lambda x, y, label: plt.plot(x, y, label=label)
# plot_fun = lambda x,y,label : plt.plot(x,y)
# plot_fun = lambda x,y,label : plt.plot(x,y,label=label,marker='x')
# plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label)

# plot_fun = lambda x,y,label : plt.scatter(x,y,label=label,s=10,marker="x",alpha=0.4)
# plot_fun = lambda x,y,label : plt.scatter(x,y,label=label)
# save_path = "/groups/sxs/hchaudha/rough/high_acc_plots/"
# save_path = "/groups/sxs/hchaudha/rough/plots/"
# save_path = "/home/hchaudha/notes/spec_accuracy/figures/"
# save_path = "/home/hchaudha/notes/spec_accuracy/L5_comparisons/"
# save_path = "/home/hchaudha/notes/spec_accuracy/L5_comparisons/L15_no_tol/"
legend_dict = {}
for key in runs_data_dict.keys():
    legend_dict[key] = None

# legend_dict = {
#     'high_accuracy_L1_main':"Old Level 1",
#     'high_accuracy_L2_main':"Old Level 2",
#     'high_accuracy_L3_main':"Old Level 3",
#     'high_accuracy_L4_main':"Old Level 4",
#     'high_accuracy_L5_main':"Old Level 5",
#     '6_set1_L6s1':'New Level 1',
#     '6_set1_L6s2':'New Level 2',
#     '6_set1_L6s3':'New Level 3',
#     '6_set1_L6s4':'New Level 4',
#     '6_set1_L6s5':'New Level 5',
#     'high_accuracy_L1':"New Level 1",
#     'high_accuracy_L2':"New Level 2",
#     'high_accuracy_L3':"New Level 3",
#     'high_accuracy_L4':"New Level 4",
#     'high_accuracy_L5':"New Level 5",
#  }


# take_abs,modification_function = True,noise_function
# take_abs,modification_function = False,noise_function
# take_abs,modification_function = True,lambda x, y, df, y_axis: noise_function(x, y, df, y_axis, scipy_or_np='np', window=10)
# take_abs,modification_function = False,derivative_function
# take_abs,modification_function = True,compute_center
# take_abs,modification_function = False,compute_center
# take_abs,modification_function = False,compute_dt_center
# take_abs,modification_function = True,min_max_r_ratio
# take_abs,modification_function = False, lambda x, y, df, y_axis: get_drift_from_center(x, y, df, y_axis)
# take_abs,modification_function = True, lambda x, y, df, y_axis: index_constraints_norm(x, y, df, y_axis, index_num=3,norm="Linf")
# take_abs,modification_function = True, lambda x, y, df, y_axis: get_num_points_subdomains(x, y, df, y_axis, regex_for_sd=r".*", print_sd_names=False)
# take_abs,modification_function = True, lambda x, y, df, y_axis: get_num_points_subdomains(x, y, df, y_axis, regex_for_sd=r'^(?!.*SphereC).*', print_sd_names=False)
# take_abs,modification_function = True, lambda x, y, df, y_axis: get_num_points_subdomains(x, y, df, y_axis, regex_for_sd=r'SphereC.*', print_sd_names=False)

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

    #   plt.title("")
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
    save_name = "Deriv_ArealMass_L16_main.png"

    save_name = Path(f"{save_report_base_folder}/{save_name}")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()
