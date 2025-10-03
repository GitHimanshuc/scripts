from pathlib import Path
import shutil
import subprocess
import re


def replace_current_file(file_path, original_str, replaced_str):
    matched_strings = []

    def callback(match):
        matched_strings.append(match.group(0))  # Store the matched string
        return replaced_str  # Replace with the new string

    with open(file_path, "r") as file:
        data = file.read()

    # Use re.sub with callback to replace and collect matched strings
    data, replaced_status = re.subn(original_str, callback, data)

    if replaced_status != 0:
        print(f"""
Replaced in File: {file_path}
Original String: {original_str}
Replaced String: {replaced_str}
Matched Strings: {matched_strings}
""")
    else:
        raise Exception(f"""
!!!!FAILED TO REPLACE!!!!
File path: {file_path}
Original String: {original_str}
Replaced String: {replaced_str}
""")

    with open(file_path, "w") as file:
        file.write(data)


def prepare_f128_run_from_checkpoint(
    source_segment_path,
    target_Ev_parent_dir_path,
    final_time,
    observe_delta_N,
    spec_home,
    checkpointer_converter_script_path,
    MakeNextSegmentScript_path,
):
    if not checkpointer_converter_script_path.exists():
        raise ValueError(
            f"Checkpointer converter script path {checkpointer_converter_script_path} does not exist"
        )
    if not source_segment_path.exists():
        raise ValueError(f"Segment path {source_segment_path} does not exist")
    if not target_Ev_parent_dir_path.exists():
        raise ValueError(f"Target path {target_Ev_parent_dir_path} does not exist")

    Ev_path = target_Ev_parent_dir_path / "Ev"
    # Ev_path.mkdir(exist_ok=False)

    # copy segment to target dir
    shutil.copytree(
        source_segment_path,
        Ev_path / source_segment_path.name,
        dirs_exist_ok=False,
        symlinks=False,
    )

    # new segment path
    base_segment_path = Ev_path / source_segment_path.name

    # rename the checkpoints dirs to Checkpoints_old
    shutil.move(
        base_segment_path / "Run" / "Checkpoints",
        base_segment_path / "Run" / "Checkpoints_old",
    )
    old_checkpoints_path = base_segment_path / "Run" / "Checkpoints_old"

    # find all the checkpoints dirs in the new segment path
    checkpoints_dirs = list(old_checkpoints_path.glob("*"))
    if len(checkpoints_dirs) == 0:
        raise ValueError(f"No checkpoints dirs found in {base_segment_path}")

    # checkpoint dir names are the num steps, sort to get the latest
    checkpoints_dirs = sorted(checkpoints_dirs, key=lambda x: int(x.name))
    print(
        f"Found {len(checkpoints_dirs)} checkpoints dirs, latest is {checkpoints_dirs[-1]}"
    )

    # create new Checkpoints dir
    new_checkpoint_path = base_segment_path / "Run" / "Checkpoints"
    new_checkpoint_path.mkdir(exist_ok=False)

    # convert the latest checkpoint to float128 and save in new Checkpoints dir
    latest_checkpoint_dir = checkpoints_dirs[-1]
    print(
        f"Converting latest checkpoint {latest_checkpoint_dir} to float128 and saving in {new_checkpoint_path}"
    )
    cmd = f"python {checkpointer_converter_script_path} {latest_checkpoint_dir} {base_segment_path / 'Run' / 'Checkpoints'}"
    print(cmd)

    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise ValueError(
            f"Checkpointer conversion failed with return code {res.returncode}"
        )

    # Before running MakeNextSegment, create a file called DisablePreArchiveBbhChecks.input
    with open(Ev_path / "DisablePreArchiveBbhChecks.input", "w") as f:
        f.write("")

    # change the working dir to Ev dir and run MakeNextSegment to create a new segment
    cmd = f"cd {Ev_path} && {MakeNextSegmentScript_path} -d {base_segment_path} -E {spec_home / 'Evolution/Executables/EvolveHyperbolicSystem'}"
    print(cmd)

    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise ValueError(f"MakeNextSegment failed with return code {res.returncode}")

    new_segment_path = sorted(list(Ev_path.glob("Lev*")))[-1]

    # match the line `TensorYlmDataBaseDir = /groups/sxs/hchaudha/.TensorYlmDB;` and replace with `TensorYlmDataBaseDir = ;`
    replace_current_file(
        new_segment_path / "Evolution.input",
        r"(TensorYlmDataBaseDir\s*=\s*)[^\n]*",
        r"\1;",
    )

    # change to FromStep restart
    replace_current_file(
        new_segment_path / "Evolution.input",
        r"(Restart    =\s*)[^\n]*",
        f"FromStep(FilenamePrefix={new_checkpoint_path};Step={latest_checkpoint_dir.stem};)",
    )

    # set final time
    replace_current_file(
        new_segment_path / "Evolution.input",
        r"FinalTime\s*=\s*\d+;",
        f"FinalTime = {final_time};",
    )

    # overwrite stuff that fails from GrWaveExtraction.input
    replace_current_file(
        new_segment_path / "GrWaveExtraction.input",
        r"(RadiiForCCE\s*=\s*)[^\n]*",
        "",
    )
    replace_current_file(
        new_segment_path / "GrWaveExtraction.input",
        r"(OutputCceTensors\s*=\s*)[^\n]*",
        "",
    )
    replace_current_file(
        new_segment_path / "GrWaveExtraction.input",
        r"(OutputCceScalars\s*=\s*)[^\n]*",
        "OutputCceScalars = no;",
    )
    replace_current_file(
        new_segment_path / "GrWaveExtraction.input",
        r"(OutputPsi3\s*=\s*)[^\n]*",
        "OutputPsi3 = no;",
    )
    replace_current_file(
        new_segment_path / "GrWaveExtraction.input",
        r"(OutputPsi2\s*=\s*)[^\n]*",
        "OutputPsi2 = no;",
    )
    replace_current_file(
        new_segment_path / "GrWaveExtraction.input",
        r"(OutputPsi1\s*=\s*)[^\n]*",
        "OutputPsi1 = no;",
    )
    replace_current_file(
        new_segment_path / "GrWaveExtraction.input",
        r"(OutputPsi0\s*=\s*)[^\n]*",
        "OutputPsi0 = no;",
    )

    # in all the input files change EveryDeltaT to EveryNSteps
    for input_file in new_segment_path.glob("*.input"):
        # Skip GrVolumeDumps.input
        if input_file.name == "GrVolumeDumps.input":
            continue
        # check if EveryDeltaT is present
        if "Ringdown" not in input_file.name:
            with open(input_file, "r") as f:
                data = f.read()
                if "EveryDeltaT" not in data:
                    continue
            replace_current_file(
                input_file,
                r"EveryDeltaT\(DeltaT=[0-9.]+\)",
                f"EveryNSteps = {observe_delta_N};",
            )
        else:
            with open(input_file, "r") as f:
                data = f.read()
                if "EveryDeltaT" not in data:
                    continue
            replace_current_file(
                input_file,
                r"EveryDeltaT\(.*?\)",
                f"EveryNSteps = {observe_delta_N}",
            )
            # append at __DeltaTObserve__ ,__TstartObserve__ at the end of the file to keep DoMultipleRuns happy
            with open(input_file, "a") as f:
                f.write("\n__DeltaTObserve__ , __TstartObserve__\n")




spec_home = Path("/home/hchaudha/spec")

final_time = 20000.0
observe_delta_N = 25

checkpointer_converter_script_path = spec_home / "convert_double_to_dd.py"
MakeNextSegmentScript_path = spec_home / "Support/bin/MakeNextSegment"


source_segment_path = Path(
    "/resnick/groups/sxs/hchaudha/spec_runs/42_f128_from_check/L3/Ev/Lev3_AE"
)
target_Ev_parent_dir_path = Path(
    "/resnick/groups/sxs/hchaudha/spec_runs/42_f128_from_check"
)

# prepare_f128_run_from_checkpoint(
#     source_segment_path,
#     target_Ev_parent_dir_path,
#     final_time,
#     observe_delta_N,
#     spec_home,
#     checkpointer_converter_script_path,
#     MakeNextSegmentScript_path,
# )
