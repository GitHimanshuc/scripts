# %%
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

def create_WT_data_compressed(
    input_file,
    output_file,
    trange=np.arange(0, 10000, 0.1),
    rand_amp=None,
    extraction_radii=100,
):
    shutil.copy(input_file, output_file)
    with h5py.File(output_file, "r+") as outfile:
        for key1, item1 in outfile.items():
            if "Version" in key1:
                continue
            if isinstance(item1, h5py.Dataset):
                print(f"----Modifying {key1}")
                data = np.zeros((len(trange), outfile[key1].shape[1]))
                print(data.shape)
                data[:, 0] = trange
                R_val = 3.544907701811031 * extraction_radii
                if "W.dat" == key1:
                    data[:, 1] = -89.09324794930309 / R_val**2
                if "R.dat" == key1:
                    data[:, 1] = R_val
                if rand_amp is not None:
                    data[:, 1:] += rand_amp * (
                        2 * np.random.rand(*data[:, 1:].shape) - 1
                    )

                # Save all attributes
                attr_dict = {}
                for attr_key, attr_value in outfile[key1].attrs.items():
                    attr_dict[attr_key] = attr_value

                del outfile[key1]
                ds = outfile.create_dataset(
                    key1,
                    data=data,
                    compression="gzip",  # or 'lzf', 'szip'
                    compression_opts=9 if "gzip" else None,  # level 0-9 for gzip
                    chunks=True,  # chunking is required for compression
                )

                # Restore all attributes
                for attr_key, attr_value in attr_dict.items():
                    ds.attrs[attr_key] = attr_value

# %%
def make_config_file(
    BoundaryDataPath: Path,
    InputSavePath: Path = None,
    which_ID: str = "ConformalFactor",
    options_dict_user={},
    observer_vol_data = False,
) -> Path:
    options_dict = {
        "Cce.Evolution.TimeStepper.AdamsBashforth.Order": 3,
        "Cce.Evolution.StepChoosers.Constant": 0.1,
        "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 1e-9,
        "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-7,
        "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 1e-9,
        "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-8,
        "Cce.LMax": 25,
        "Cce.NumberOfRadialPoints": 20,
        "Cce.ObservationLMax": 8,
        "Cce.H5Interpolator.BarycentricRationalSpanInterpolator.MinOrder": 10,
        "Cce.H5Interpolator.BarycentricRationalSpanInterpolator.MaxOrder": 10,
        "Cce.H5LookaheadTimes": 10000,
        "Cce.Filtering.RadialFilterHalfPower": 64,
        "Cce.Filtering.RadialFilterAlpha": 35.0,
        "Cce.Filtering.FilterLMax": 18,
        "Cce.ScriInterpOrder": 5,
        "Cce.ScriOutputDensity": 1,
    }

    for key in options_dict_user.keys():
        if key not in options_dict:
            raise ValueError(f"Key {key} is not a valid option.")
        else:
            options_dict[key] = options_dict_user[key]

    CCE_ID_data = ""
    match which_ID:
        case "ConformalFactor":
            CCE_ID_data = """
    ConformalFactor:
      AngularCoordTolerance: 1e-13
      MaxIterations: 1000 # Do extra iterations in case we improve.
      RequireConvergence: False # Often don't converge to 1e-13, but that's fine
      OptimizeL0Mode: True
      UseBetaIntegralEstimate: False
      ConformalFactorIterationHeuristic: SpinWeight1CoordPerturbation
      UseInputModes: False
      InputModes: []
"""
        case "InverseCubic":
            CCE_ID_data = """
    InverseCubic:
"""
        case "ZeroNonSmooth":
            CCE_ID_data = """
    ZeroNonSmooth:
      AngularCoordTolerance: 1e-13
      MaxIterations: 1000
      RequireConvergence: False
"""
        case "NoIncomingRadiation":
            CCE_ID_data = """
    NoIncomingRadiation:
      AngularCoordTolerance: 1e-13
      MaxIterations: 1000
      RequireConvergence: False
"""
    if InputSavePath is None:
        InputSavePath = BoundaryDataPath.parent / "cce.yaml"
    assert InputSavePath.parent.exists()

    config_file = f"""
# Distributed under the MIT License.
# See LICENSE.txt for details.

# This block is used by testing and the SpECTRE command line interface.
Executable: CharacteristicExtract
Testing:
  Check: parse
  Priority: High

---
Evolution:
  InitialTimeStep: 0.25
  MinimumTimeStep: 1e-7
  InitialSlabSize: 10.0

ResourceInfo:
  AvoidGlobalProc0: false
  Singletons: Auto

Observers:
  VolumeFileName: "vol_{str(InputSavePath.stem)}"
  ReductionFileName: "red_{str(InputSavePath.stem)}"

EventsAndTriggersAtSlabs:
  # Write the CCE time step every Slab. A Slab is a fixed length of simulation
  # time and is not influenced by the dynamically adjusted step size.
  - Trigger:
      Slabs:
        EvenlySpaced:
          Offset: 0
          Interval: 1
    Events:
      - ObserveTimeStep:
          # The output is written into the "ReductionFileName" HDF5 file under
          # "/SubfileName.dat"
          SubfileName: CceTimeStep
          PrintTimeToTerminal: true
      - ObserveFields:
          VariablesToObserve:
            - BondiBeta
            - Du(J)
            - DuRDividedByR
            - Dy(BondiBeta)
            - Dy(Du(J))
            - Dy(Dy(BondiBeta))
            - Dy(Dy(Du(J)))
            - Dy(Dy(J))
            - Dy(Dy(Q))
            - Dy(Dy(U))
            - Dy(Dy(W))
            - Dy(H)
            - Dy(J)
            - Dy(Q)
            - Dy(U)
            - Dy(W)
            - EthRDividedByR
            - H
            - InertialRetardedTime
            - J
            - OneMinusY
            - Psi0
            - Psi1
            - Q
            - R
            - U
            - W

EventsAndTriggersAtSteps:

Cce:
  Evolution:
    TimeStepper:
      AdamsBashforth:
        Order: {options_dict['Cce.Evolution.TimeStepper.AdamsBashforth.Order']} # Going to higher order doesn't seem necessary for CCE
    StepChoosers:
      - Constant: {options_dict['Cce.Evolution.StepChoosers.Constant']} # Don't take steps bigger than 0.1M
      - LimitIncrease:
          Factor: 2
      - ErrorControl(SwshVars):
          AbsoluteTolerance: {options_dict['Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance']}
          RelativeTolerance: {options_dict['Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance']}
          # These factors control how much the time step is changed at once.
          MaxFactor: 2
          MinFactor: 0.25
          # How close to the "perfect" time step we take. Since the "perfect"
          # value assumes a linear system, we need some safety factor since our
          # system is nonlinear, and also so that we reduce how often we retake
          # time steps.
          SafetyFactor: 0.9
      - ErrorControl(CoordVars):
          AbsoluteTolerance: {options_dict['Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance']}
          RelativeTolerance: {options_dict['Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance']}
          # These factors control how much the time step is changed at once.
          MaxFactor: 2
          MinFactor: 0.25
          # How close to the "perfect" time step we take. Since the "perfect"
          # value assumes a linear system, we need some safety factor since our
          # system is nonlinear, and also so that we reduce how often we retake
          # time steps.
          SafetyFactor: 0.9

  # The number of angular modes used by the CCE evolution. This must be larger
  # than ObservationLMax. We always use all of the m modes for the LMax since
  # using fewer m modes causes aliasing-driven instabilities.
  LMax: {options_dict['Cce.LMax']}
  # Probably don't need more than 15 radial grid points, but could increase
  # up to ~20
  NumberOfRadialPoints: {options_dict['Cce.NumberOfRadialPoints']}
  # The maximum ell we use for writing waveform output. While CCE can dump
  # more, you should be cautious with higher modes since mode mixing, truncation
  # error, and systematic numerical effects can have significant contamination
  # in these modes.
  ObservationLMax: {options_dict['Cce.ObservationLMax']}

  InitializeJ:
    # To see what other J-initialization procedures are available, comment
    # out this group of options and do, e.g. "Blah:" The code will print
    # an error message with the available options and a help string.
    # More details can be found at spectre-code.org.
{CCE_ID_data}

  StartTime: Auto
  EndTime: Auto
  ExtractionRadius: Auto

  BoundaryDataFilename: {BoundaryDataPath.name}
  H5Interpolator:
    BarycentricRationalSpanInterpolator:
      MinOrder: {options_dict['Cce.H5Interpolator.BarycentricRationalSpanInterpolator.MinOrder']}
      MaxOrder: {options_dict['Cce.H5Interpolator.BarycentricRationalSpanInterpolator.MaxOrder']}

  H5LookaheadTimes: {options_dict['Cce.H5LookaheadTimes']}

  Filtering:
    RadialFilterHalfPower: {options_dict['Cce.Filtering.RadialFilterHalfPower']}
    RadialFilterAlpha: {options_dict['Cce.Filtering.RadialFilterAlpha']}
    FilterLMax: {options_dict['Cce.Filtering.FilterLMax']}

  ScriInterpOrder: {options_dict['Cce.ScriInterpOrder']}
  ScriOutputDensity: {options_dict['Cce.ScriOutputDensity']}

"""

    with InputSavePath.open("w") as f:
        f.writelines(config_file)

    return InputSavePath


def make_submit_file(
    save_folder_path: Path,
    cce_input_file_path: Path,
    CCE_Executable_path: Path,
    write_scripts_only=False,
):
    submit_script = f"""#!/bin/bash -
#SBATCH -J CCE_{save_folder_path.stem}             # Job Name
#SBATCH -o CCE.stdout                 # Output file name
#SBATCH -e CCE.stderr                 # Error file name
#SBATCH -n 2                          # Number of cores
#SBATCH -p expansion                  # Queue name
#SBATCH --ntasks-per-node 2           # number of MPI ranks per node
#SBATCH -t 24:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue
#SBATCH --reservation=sxs_standing

# Spectre related stuff to load JeMalloc
export SPECTRE_HOME=/central/groups/sxs/hchaudha/spectre
. $SPECTRE_HOME/support/Environments/caltech_hpc_gcc.sh
spectre_load_modules

# Go to the correct folder with the boundary data
cd {save_folder_path}

# run CCE
{CCE_Executable_path} --input-file ./{cce_input_file_path.name}
"""
    submit_script_path = save_folder_path / "submit.sh"
    submit_script_path.write_text(submit_script)

    if not write_scripts_only:
        command = f"cd {save_folder_path} && qsub {submit_script_path}"
        status = subprocess.run(command, capture_output=True, shell=True, text=True)
        if status.returncode == 0:
            print(f"Succesfully submitted {submit_script_path}\n{status.stdout}")
        else:
            sys.exit(
                f"Job submission failed for {submit_script_path} with error: \n{status.stdout} \n{status.stderr}"
            )

# %%
def create_CCE_single_bh(
    NewCCEPath: Path,
    BondiBaseH5File: Path,
    CCE_Executable_path: Path,
    ExtractionRadii_str: str,
    CCE_ID: str = "ConformalFactor",
    options_dict_user={},
    write_scripts_only=False,
    trange=np.arange(0, 10000, 0.1),
    rand_amp=None,
):
    NewCCEPath = NewCCEPath.resolve()
    BondiBaseH5File = BondiBaseH5File.resolve()
    CCE_Executable_path = CCE_Executable_path.resolve()

    if not BondiBaseH5File.exists():
        raise Exception(f"{BondiBaseH5File} does not exist!")
    if not CCE_Executable_path.exists():
        raise Exception(f"{CCE_Executable_path} does not exist!")

    NewCCEPath.mkdir(parents=True, exist_ok=False)

    create_WT_data_compressed(
        BondiBaseH5File,
        NewCCEPath / f"BondiDataR{ExtractionRadii_str}.h5",
        trange=trange,
        rand_amp=rand_amp,
        extraction_radii=int(ExtractionRadii_str),
    )

    make_config_file(
        BoundaryDataPath=NewCCEPath / f"BondiDataR{ExtractionRadii_str}.h5",
        InputSavePath=NewCCEPath / f"cce.yaml",
        which_ID=CCE_ID,
        options_dict_user=options_dict_user,
    )

    make_submit_file(
        save_folder_path=NewCCEPath,
        cce_input_file_path=NewCCEPath / f"cce.yaml",
        CCE_Executable_path=CCE_Executable_path,
        write_scripts_only=write_scripts_only,
    )

    print("DONE!\n\n\n\n\n")


# for L,R in itertools.product([25,20,15,10,5], [20,15,10,5]):
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/test1/LMax{L}_RadPts{R}"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str='0100',
#         CCE_ID="ConformalFactor",
#         trange=np.arange(0, 5000, 0.1),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 1e-14,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 1e-14,
#             "Cce.LMax": L,
#             "Cce.NumberOfRadialPoints": R,
#         }
#     )


# for ex_rad in ['0012', '0050', '0112', '0200', '0312', '0450', '0612', '0800', '1012', '1250', '1512', '1800', '2112', '2450', '2812', '3200', '3612', '4050', '4512', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0015', '0020', '0030', '0075']:
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/radius_dependence/ex_rad_{ex_rad}"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="ConformalFactor",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 1e-13,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 1e-13,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )

# for ex_rad in ['0010', '0025', '0050', '0100', '0250', '0500' '0750', '1000', '1500', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000']:
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/new_exe_rad_dep/ex_rad_{ex_rad}"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="ConformalFactor",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 1e-13,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 1e-13,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )

# for abs_tol, ex_rad in itertools.product([10,11,12,13,14],['0100','0500', '1000', '2500', '5000', '9999']):
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/rad_ode_tol_dep/rad_{ex_rad}_abstol_{abs_tol}"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="ConformalFactor",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )


# for abs_tol, ex_rad in itertools.product([15,16,17,18],['0100','0500', '1000', '2500', '5000', '9999']):
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/rad_ode_tol_dep/rad_{ex_rad}_abstol_{abs_tol}"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="ConformalFactor",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )


# for abs_tol, ex_rad in itertools.product([15],['0100','0500', '1000', '2500', '5000', '9999']):
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/obs_vol_data/rad_{ex_rad}"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="ConformalFactor",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )



# for abs_tol, ex_rad in itertools.product([15],['0500', '2500']):
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/obs_vol_data/rad_{ex_rad}_IC"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="InverseCubic",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )


# for abs_tol, ex_rad in itertools.product([15],['0500', '2500']):
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/obs_vol_data/rad_{ex_rad}_ZNS"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="ZeroNonSmooth",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )


# for abs_tol, ex_rad in itertools.product([15],['0500', '2500']):
#     create_CCE_single_bh(
#         NewCCEPath = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/obs_vol_data/rad_{ex_rad}_NIR"),
#         BondiBaseH5File = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/BondiCceR0100.h5"),
#         CCE_Executable_path = Path("/groups/sxs/hchaudha/spec_runs/single_bh_CCE/data/CharacteristicExtract"),
#         ExtractionRadii_str=ex_rad,
#         CCE_ID="NoIncomingRadiation",
#         trange=np.arange(0, 100000, 0.5),
#         options_dict_user = {
#             "Cce.Evolution.StepChoosers.Constant": 100,
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(SwshVars).RelativeTolerance": 1e-12,
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).AbsoluteTolerance": 10**(-abs_tol),
#             "Cce.Evolution.StepChoosers.ErrorControl(CoordVars).RelativeTolerance": 1e-12,
#             "Cce.LMax": 10,
#             "Cce.NumberOfRadialPoints": 10,
#         },
#     )