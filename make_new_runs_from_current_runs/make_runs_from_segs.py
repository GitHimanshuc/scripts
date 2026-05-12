# # Script part


import json
from pathlib import Path
import shutil
import re
import subprocess
import sys


spec_home = "/home/hchaudha/spec"


def check_regex_in_file(file_path, regex_pattern):
    # Read the content of the file
    with open(file_path, "r") as file:
        content = file.read()

    # Try to compile the regex pattern
    try:
        pattern = re.compile(regex_pattern)
    except re.error:
        print("Invalid regex pattern.")
        return False

    # Search for matches
    if pattern.search(content):
        return True
    else:
        return False

def return_first_regex_match_in_file(file_path, regex_pattern):
    with open(file_path, "r") as file:
        content = file.read()

    try:
        pattern = re.compile(regex_pattern)
    except re.error:
        print("Invalid regex pattern.")
        return None

    match = pattern.search(content)

    if match:
        return match.group(0)   # matched string
    else:
        return None

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


def is_continuous(lst: list):
    # my_list = ["AZ", "BA", "BB", "BC"]
    # print(is_continuous(my_list))  # Should return True

    # my_list = ["AA", "AB", "AC", "AD"]
    # print(is_continuous(my_list))  # Should return True

    # my_list = ["AA", "AB", "AD", "AC"]
    # print(is_continuous(my_list))  # Should return False

    for i in range(len(lst) - 1):
        current = lst[i]
        next_elem = lst[i + 1]

        # Determine the next expected element
        if current[1] == "Z":
            # If the second character is Z, the next expected should start the next sequence
            expected_next_first = chr(ord(current[0]) + 1)
            expected_next_second = "A"
        else:
            # Otherwise, continue by incrementing the second character
            expected_next_first = current[0]
            expected_next_second = chr(ord(current[1]) + 1)

        expected_next_elem = expected_next_first + expected_next_second

        # If the next element is not the expected one, return False
        if next_elem != expected_next_elem:
            return False

    return True  # All elements are continuous


def verify_data_dict(data_dict: dict):
    for new_folder_name in data_dict.keys():
        data = data_dict[new_folder_name]
        if not data["new_run_parent"].exists():
            raise Exception(f"""{data['new_run_parent']=} does not exist!""")
        if not data["old_Ev_path"].exists():
            raise Exception(f"""{data['old_Ev_path']=} does not exist!""")

        if data["Ev_is_present"]:
            if data["copy_ID"]:
                raise Exception(
                    f"When Ev_is_present is set to true copy_ID and copy_bin should be false."
                )
            if data["copy_bin"]:
                raise Exception(
                    f"When Ev_is_present is set to true copy_ID and copy_bin should be false."
                )

            Ev_parent_folder_path = data["new_run_parent"] / new_folder_name
            if not Ev_parent_folder_path.exists():
                raise Exception(
                    f"Ev_is_present is set to true but {Ev_parent_folder_path=} does not exist."
                )

            Ev_folder_path = Ev_parent_folder_path / "Ev"
            if not Ev_folder_path.exists():
                raise Exception(
                    f"Ev_is_present is set to true but {Ev_folder_path=} does not exist."
                )

        if data["copy_ID"]:
            old_ID_path = data["old_Ev_path"].parent / "ID"
            if not old_ID_path.exists():
                raise Exception(
                    f"ID folder marked for copy but {old_ID_path=} does not exist."
                )

        if data["copy_bin"]:
            old_bin_path = data["old_Ev_path"] / "bin"
            if not old_bin_path.exists():
                raise Exception(
                    f"bin folder marked for copy but {old_bin_path=} does not exist."
                )

        levs_to_copy = data["levs_to_copy"]
        for lev_dict in levs_to_copy:
            lev_number = int(lev_dict["new_Lev_name"][3:])
            segments_to_copy = lev_dict["segments_to_copy"]

            # Check that the segments are continuous and sorted
            if not is_continuous(segments_to_copy):
                raise Exception(f"{segments_to_copy=} is not a continuous list!")
            
            # If using_constant_AMR_tol is False then we should have my $UseSpatiallyConstantAMRTolerances = 0; in DoMultipleRuns.input
            last_segment_path_DoMultipleRuns = data["old_Ev_path"] / (lev_dict["old_Lev_name"] + "_" + segments_to_copy[-1]) / "Run" / "DoMultipleRuns.input"
            if int(return_first_regex_match_in_file(last_segment_path_DoMultipleRuns, r"UseSpatiallyConstantAMRTolerances\s*=\s*(\d);").split()[-1][:-1]) == 1 and not lev_dict["using_constant_AMR_tol"]:
                raise Exception(f"{last_segment_path_DoMultipleRuns=} has UseSpatiallyConstantAMRTolerances set to 1 but using_constant_AMR_tol is set to False. These should be consistent.")
            if int(return_first_regex_match_in_file(last_segment_path_DoMultipleRuns, r"UseSpatiallyConstantAMRTolerances\s*=\s*(\d);").split()[-1][:-1]) == 0 and lev_dict["using_constant_AMR_tol"]:
                raise Exception(f"{last_segment_path_DoMultipleRuns=} has UseSpatiallyConstantAMRTolerances set to 0 but using_constant_AMR_tol is set to True. These should be consistent.")


            if lev_dict["this_lev_is_continuation"]:
                # Make sure that the new lev name and the old lev name are the same
                if lev_dict["old_Lev_name"] != lev_dict["new_Lev_name"]:
                    raise Exception(
                        f'{lev_dict["this_lev_is_continuation"]=} is set to true but the old lev name {lev_dict["old_Lev_name"]=} is different from the new one {lev_dict["new_Lev_name"]=}.'
                    )

                if ("new_Lev_for_AMR_tolerance" in lev_dict) or (
                    "new_Lev_for_GrDomain_and_AmrDriver" in lev_dict
                ):
                    raise Exception(
                        f'{lev_dict["this_lev_is_continuation"]=} is set to true but keys new_Lev_for_AMR_tolerance or new_Lev_for_GrDomain_and_AmrDriver are defined!'
                    )

                if "AH_factor" in lev_dict:
                    raise Exception(
                        f'{lev_dict["this_lev_is_continuation"]=} is set to true but key AH_factor is defined!'
                    )

            else:
                # Check the values of new new_Lev_for_AMR_tolerance and new_Lev_for_GrDomain_and_AmrDriver
                new_Lev_for_AMR_tolerance = lev_dict["new_Lev_for_AMR_tolerance"]
                new_Lev_for_GrDomain_and_AmrDriver = lev_dict[
                    "new_Lev_for_GrDomain_and_AmrDriver"
                ]
                # TODO deal with 45 55 etc as 4.5  5.5
                if (isinstance(new_Lev_for_AMR_tolerance, int)) or (
                    isinstance(new_Lev_for_AMR_tolerance, float)
                ):
                    pass
                else:
                    raise Exception(
                        f"{new_Lev_for_AMR_tolerance=} should be a float or an int."
                    )

                if not isinstance(new_Lev_for_GrDomain_and_AmrDriver, int):
                    raise Exception(
                        f"{new_Lev_for_GrDomain_and_AmrDriver=} should be an int."
                    )
                # If this_lev_is_continuation is False and levs have same name then print a warning
                if lev_dict["old_Lev_name"] == lev_dict["new_Lev_name"]:
                    if (
                        data["new_run_parent"] / new_folder_name
                        == data["old_Ev_path"].parent
                    ):
                        print(
                            f'\nWARNING!!\n {lev_dict["this_lev_is_continuation"]=} is set to false but the new lev name {lev_dict["old_Lev_name"]=} is same as the old one {lev_dict["new_Lev_name"]=}.'
                        )

            for folder in segments_to_copy:
                # Check that the levs we want to copy exist
                old_folder_path = data["old_Ev_path"] / (
                    lev_dict["old_Lev_name"] + "_" + folder
                )
                if not old_folder_path.exists():
                    raise Exception(f"{old_folder_path=} does not exist")

                # Check that the files being replace exist
                if folder == segments_to_copy[-1]:
                    files_to_change_in_the_new_lev = lev_dict[
                        "files_to_change_in_the_new_lev"
                    ]
                    for file_name in files_to_change_in_the_new_lev:
                        file_path = old_folder_path / file_name
                        if not file_path.exists():
                            raise Exception(
                                f"{file_path=} is supposed to be change in the new lev but it does not exist in the old folders lev."
                            )

                        # Check that the lists original_str and replaced_str have the same length
                        original_str = files_to_change_in_the_new_lev[file_name][
                            "original_str"
                        ]
                        replaced_str = files_to_change_in_the_new_lev[file_name][
                            "replaced_str"
                        ]
                        if len(original_str) != len(replaced_str):
                            raise Exception(
                                f"{original_str=} and {replaced_str=} have different lengths!"
                            )
                        for string_to_be_replaced in original_str:
                            if not check_regex_in_file(
                                file_path, string_to_be_replaced
                            ):
                                raise Exception(
                                    f"{string_to_be_replaced=} not found in the file {file_path=}!"
                                )


def MakeNextSegment(EV_folder_path: Path, previous_segment_path: Path):
    command = f"cd {EV_folder_path} && {spec_home}/Support/bin/MakeNextSegment -d {previous_segment_path} -t . -S"
    status = subprocess.run(command, capture_output=True, shell=True, text=True)
    if status.returncode == 0:
        print(
            f"Succesfully ran MakeNextSegment in {EV_folder_path}: \n {status.stdout}"
        )
    else:
        sys.exit(
            f"MakeNextSegment failed in {EV_folder_path} with error: \n {status.stdout} \n {status.stderr}"
        )


def get_next_segment(lst):
    # my_list = ["AA", "AB", "AC", "AD", "BA"]
    # print(get_next_segment(my_list))  # Should return "BB"

    # my_list = ["AZ"]
    # print(get_next_segment(my_list))  # Should return "BA"

    # my_list = []
    # print(get_next_segment(my_list))  # Should return "AA"

    if not lst:
        return "AA"  # Return "AA" if the list is empty

    # Sort the list to ensure that elements are in the correct order
    lst.sort()

    # Retrieve the last element from the sorted list
    last_elem = lst[-1]

    # Determine the next element
    if last_elem[1] == "Z":
        # If the second character is 'Z', increment the first character and set the second to 'A'
        next_first = chr(ord(last_elem[0]) + 1)
        next_second = "A"
    else:
        # Otherwise, simply increment the second character
        next_first = last_elem[0]
        next_second = chr(ord(last_elem[1]) + 1)

    next_elem = next_first + next_second
    return next_elem


def link_or_copy(source_path: Path, destination_path: Path, link_or_copy_folders: str):
    if link_or_copy_folders == "link":
        # Create a symbolic link
        if not destination_path.exists():
            destination_path.symlink_to(source_path)
            print(f"Created a symbolic link from {source_path} to {destination_path}")
        else:
            print(
                f"Destination {destination_path} already exists. Cannot create a link."
            )

    elif link_or_copy_folders == "copy":
        # Copy the directory using shutil when using pathlib
        if not destination_path.exists():
            shutil.copytree(source_path, destination_path)
            print(f"Copied {source_path} to {destination_path}")
            # When copying add write permission as well
            add_write_permission(destination_path)
        else:
            print(f"Destination {destination_path} already exists. Cannot copy.")

    else:
        print("Invalid option. Please use 'link' or 'copy'.")


def add_write_permission(target_dir: Path):
    if not target_dir.exists():
        raise Exception(
            f"Trying to change the permissions of the folder {target_dir} which does not exist."
        )
    if target_dir.is_symlink():
        raise Exception(
            f"Trying to get write permission on a symlink folder {target_dir}. This should not be happening."
        )
    else:
        command = f"chmod u+w -R {target_dir.resolve(strict=True)}"
        status = subprocess.run(command, capture_output=True, shell=True, text=True)
        if status.returncode == 0:
            print(f"Succesfully added execute permission to {target_dir}")
        else:
            sys.exit(f"{command}\nfailed with error: \n {status.stderr}")


def copy_all_files(src: Path, dst: Path, exclude_files: list[str]):
    if not src.exists():
        raise Exception(f"{src=} does not exist")
    if not dst.exists():
        raise Exception(f"{dst=} does not exist")

    # Iterate through the source directory
    for path in src.iterdir():
        if path.is_dir():
            # If it's a directory, do not copy
            continue
        else:
            excluded_file = False
            for excluded in exclude_files:
                if excluded in str(path):
                    excluded_file = True
                    continue
            # If it's not an excluded file, copy it
            if not excluded_file:
                shutil.copy2(path, dst)


def change_AmrTolerances(
    segement_path: Path, new_lev: int | float, AH_factor: float, using_constant_AMR_tol: bool
):
    file_path = segement_path / "AmrTolerances.input"


    # If we are using UseSpatiallyConstantAMRTolerances then:
    #   if ($UseSpatiallyConstantAMRTolerances){
    #     $TruncationErrorMax = $TruncationErrorMax*1.0e-2;
    #     $AMR_GaussianAmplitude_A = 0.0;
    #     $AMR_GaussianAmplitude_B = 0.0;
    #   }
    # We just write TruncationErrorMax, A and B do not matter. AH calculations should be done with base TruncationErrorMax
    TruncationErrorMaxFactor = 1.0
    if using_constant_AMR_tol:
        TruncationErrorMaxFactor *= 1.0e-2

    TruncationErrorMax = 0.000216536 * 4 ** (-new_lev)
    replace_current_file(
        file_path,
        r"TruncationErrorMax=.*;",
        f"TruncationErrorMax={TruncationErrorMax*TruncationErrorMaxFactor};",
    )

    ProjectedConstraintsMax = 0.216536 * 4 ** (-new_lev)
    replace_current_file(
        file_path,
        r"ProjectedConstraintsMax=.*;",
        f"ProjectedConstraintsMax={ProjectedConstraintsMax};",
    )
    TruncationErrorMaxA = TruncationErrorMax * 1.0e-4
    replace_current_file(
        file_path,
        r"TruncationErrorMaxA=.*;",
        f"TruncationErrorMaxA={TruncationErrorMaxA};",
    )
    TruncationErrorMaxB = TruncationErrorMax * 1.0e-4
    replace_current_file(
        file_path,
        r"TruncationErrorMaxB=.*;",
        f"TruncationErrorMaxB={TruncationErrorMaxB};",
    )

    AhMaxRes = TruncationErrorMax / AH_factor
    replace_current_file(file_path, r"AhMaxRes=.*;", f"AhMaxRes={AhMaxRes};")
    AhMinRes = AhMaxRes / 10.0
    replace_current_file(file_path, r"AhMinRes=.*;", f"AhMinRes={AhMinRes};")

    AhMaxTrunc = TruncationErrorMax / AH_factor
    replace_current_file(file_path, r"AhMaxTrunc=.*;", f"AhMaxTrunc={AhMaxTrunc};")
    AhMinTrunc = AhMaxTrunc / 100.0
    replace_current_file(file_path, r"AhMinTrunc=.*;", f"AhMinTrunc={AhMinTrunc};")



def change_GrDomain_and_AmrDriver_Level(segement_path: Path, new_lev: int):
    AmrDriver_path = segement_path / "AmrDriver.input"
    replace_current_file(AmrDriver_path, r"Level\s?=\s?\d;", f"Level = {new_lev};")

    GrDomain_path = segement_path / "GrDomain.input"
    # There is also a WarningLevel = 0; That should not be changed
    replace_current_file(GrDomain_path, r"Level\s?=\s?\d;", f"Level = {new_lev};")


def create_new_folder(folder_name: str, data_dict: dict):
    folder_dict = data_dict[folder_name]
    new_run_parent: Path = folder_dict["new_run_parent"]
    new_run_path = new_run_parent / folder_name
    old_Ev_path: Path = folder_dict["old_Ev_path"]
    link_or_copy_folders = folder_dict["link_or_copy_folders"]
    copy_ID = folder_dict["copy_ID"]
    old_ID_path = old_Ev_path / "ID"
    copy_bin = folder_dict["copy_bin"]
    old_bin_path = old_Ev_path / "bin"

    levs_to_copy = folder_dict["levs_to_copy"]

    new_run_path.mkdir(exist_ok=True)
    new_Ev_path = new_run_path / "Ev"
    new_ID_path = new_run_path / "ID"
    new_bin_path = new_run_path / "Ev/bin"
    new_Ev_path.mkdir(exist_ok=True)
    if copy_ID:
        link_or_copy(old_ID_path, new_ID_path, link_or_copy_folders)
    if copy_bin:
        link_or_copy(old_bin_path, new_bin_path, link_or_copy_folders)

    levs_to_copy = folder_dict["levs_to_copy"]
    for lev_dict in levs_to_copy:
        segments_to_copy = lev_dict["segments_to_copy"]

        for segment in segments_to_copy:
            old_segment_path = folder_dict["old_Ev_path"] / (
                lev_dict["old_Lev_name"] + "_" + segment
            )
            new_segment_path = new_Ev_path / (lev_dict["new_Lev_name"] + "_" + segment)
            link_or_copy(old_segment_path, new_segment_path, link_or_copy_folders)

        # Get the last segment copied or linked
        last_segment_path = new_Ev_path / (
            lev_dict["new_Lev_name"] + "_" + segments_to_copy[-1]
        )
        last_segment_bin_path = (
            new_Ev_path
            / (lev_dict["new_Lev_name"] + "_" + segments_to_copy[-1])
            / "bin"
        )

        # Make the next segment
        new_segment_path = new_Ev_path / (
            lev_dict["new_Lev_name"] + "_" + get_next_segment(segments_to_copy)
        )
        new_segment_path.mkdir()
        # Copy the bin folder
        link_or_copy(last_segment_bin_path, new_segment_path / "bin", "copy")
        copy_all_files(last_segment_path, new_segment_path, ["SpEC."])

        # Add user write permission
        add_write_permission(new_segment_path)

        # Change the Restart option in the Evolution.input
        last_segment_checkpoint_folder_path = last_segment_path / "Run/Checkpoints/"
        if not last_segment_checkpoint_folder_path.exists():
            raise Exception(f"{last_segment_checkpoint_folder_path=} does not exist!")

        if lev_dict["this_lev_is_continuation"]:
            # For continuation runs just change the FromLastStep
            check_regex_in_file(
                new_segment_path / "Evolution.input",
                r"Restart\s*=\s*FromLastStep\(.*\);",
            )
            replace_current_file(
                new_segment_path / "Evolution.input",
                r"Restart\s*=\s*FromLastStep\(.*\);",
                f"Restart    = FromLastStep(FilenamePrefix={last_segment_checkpoint_folder_path}/;);",
            )
        else:
            check_regex_in_file(
                new_segment_path / "Evolution.input",
                r"Restart\s*=\s*FromLastStep\(.*\);",
            )
            replace_current_file(
                new_segment_path / "Evolution.input",
                r"Restart\s*=\s*FromLastStep\(.*\);",
                f"Restart    = FromLastStep(FilenamePrefix={last_segment_checkpoint_folder_path}/;GrGlobalVarsCheckpoint = Interpolated(FlagAllSubdomainsAsChanged = yes; ResetFilterFunctionsForAMR = yes; StartAmrWithMinimumExtents = yes; ResolutionChanger=Spectral; Interpolator =ParallelAdaptive(TopologicalInterpolator=CardinalInterpolator;); DomainFile=GrDomain.input; DomainDir={last_segment_path}/Run/););",
            )

            # Change the Submit.sh run name
            replace_current_file(
                new_segment_path / "MakeSubmit.input",
                r"Jobname.*",
                f"Jobname = {folder_name}.{lev_dict['new_Lev_name']}",
            )
            replace_current_file(
                new_segment_path / "Submit.sh",
                r"#SBATCH -J.*",
                f"#SBATCH -J {folder_name}.{lev_dict['new_Lev_name']}",
            )

            change_AmrTolerances(
                new_segment_path,
                lev_dict["new_Lev_for_AMR_tolerance"],
                lev_dict["AH_factor"],
                lev_dict["using_constant_AMR_tol"]
            )
            change_GrDomain_and_AmrDriver_Level(
                new_segment_path, lev_dict["new_Lev_for_GrDomain_and_AmrDriver"]
            )

        # Replace the text in files
        files_to_change_in_the_new_lev = lev_dict["files_to_change_in_the_new_lev"]
        for file_name in files_to_change_in_the_new_lev:
            file_path = new_segment_path / file_name
            if not file_path.exists():
                raise Exception(
                    f"{file_path=} is supposed to be change in the new segment but it does not exist."
                )

            original_str_list = files_to_change_in_the_new_lev[file_name][
                "original_str"
            ]
            replaced_str_list = files_to_change_in_the_new_lev[file_name][
                "replaced_str"
            ]

            for original_str, replaced_str in zip(original_str_list, replaced_str_list):
                replace_current_file(file_path, original_str, replaced_str)



data_dict = {
    "2000M_AHtol_10": {
        "spec_home": Path("/home/hchaudha/spec"),
        "Comment": "Test higher AH tol to see if the noise goes down.",
        "new_run_parent": Path("/resnick/groups/sxs/hchaudha/spec_runs/56_segs"),
        "old_Ev_path": Path(
            "/resnick/groups/sxs/hchaudha/spec_runs/56_lsds_large_ext_bdr_long/q1_ns_2000M_lsds_long_inspi/Ecc3/Ev"
        ),
        "link_or_copy_folders": "link",
        "Ev_is_present": False,
        "copy_ID": False,
        "copy_bin": False,
        "levs_to_copy": [
            {
                "old_Lev_name": "Lev4",
                "new_Lev_name": "Lev4",
                "using_constant_AMR_tol": True,
                "this_lev_is_continuation": False,
                "new_Lev_for_AMR_tolerance": 4,
                "new_Lev_for_GrDomain_and_AmrDriver": 4,
                "AH_factor": 10.0,
                # "segments_to_copy": ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL'],
                "segments_to_copy": [
                    "AA",
                    "AB",
                    "AC",
                    "AD",
                    "AE",
                    "AF",
                    "AG",
                    "AH",
                    "AI",
                    "AJ",
                ],
                "files_to_change_in_the_new_lev": {
                    "AmrTolerances.input": {
                        "original_str": [r"ODETolerance=([^;]*);"],
                        "replaced_str": [
                            f"ODETolerance = {0.000216536 / 2000 * 4 ** (-4)};"
                        ],
                    },
                    "Evolution.input": {
                        "original_str": [
                            r"FinalTime = \d*;",
                        ],
                        "replaced_str": [
                            "FinalTime = 35000;",
                        ],
                    },
                },
            },
        ],
    },
    "2000M_AHtol_100": {
        "spec_home": Path("/home/hchaudha/spec"),
        "Comment": "Test higher AH tol to see if the noise goes down.",
        "new_run_parent": Path("/resnick/groups/sxs/hchaudha/spec_runs/56_segs"),
        "old_Ev_path": Path(
            "/resnick/groups/sxs/hchaudha/spec_runs/56_lsds_large_ext_bdr_long/q1_ns_2000M_lsds_long_inspi/Ecc3/Ev"
        ),
        "link_or_copy_folders": "link",
        "Ev_is_present": False,
        "copy_ID": False,
        "copy_bin": False,
        "levs_to_copy": [
            {
                "old_Lev_name": "Lev4",
                "new_Lev_name": "Lev4",
                "using_constant_AMR_tol": True,
                "this_lev_is_continuation": False,
                "new_Lev_for_AMR_tolerance": 4,
                "new_Lev_for_GrDomain_and_AmrDriver": 4,
                "AH_factor": 100.0,
                # "segments_to_copy": ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL'],
                "segments_to_copy": [
                    "AA",
                    "AB",
                    "AC",
                    "AD",
                    "AE",
                    "AF",
                    "AG",
                    "AH",
                    "AI",
                    "AJ",
                ],
                "files_to_change_in_the_new_lev": {
                    "AmrTolerances.input": {
                        "original_str": [r"ODETolerance=([^;]*);"],
                        "replaced_str": [
                            f"ODETolerance = {0.000216536 / 2000 * 4 ** (-4)};"
                        ],
                    },
                    "Evolution.input": {
                        "original_str": [
                            r"FinalTime = \d*;",
                        ],
                        "replaced_str": [
                            "FinalTime = 35000;",
                        ],
                    },
                },
            },
        ],
    },
    "2000M_AHtol_1000": {
        "spec_home": Path("/home/hchaudha/spec"),
        "Comment": "Test higher AH tol to see if the noise goes down.",
        "new_run_parent": Path("/resnick/groups/sxs/hchaudha/spec_runs/56_segs"),
        "old_Ev_path": Path(
            "/resnick/groups/sxs/hchaudha/spec_runs/56_lsds_large_ext_bdr_long/q1_ns_2000M_lsds_long_inspi/Ecc3/Ev"
        ),
        "link_or_copy_folders": "link",
        "Ev_is_present": False,
        "copy_ID": False,
        "copy_bin": False,
        "levs_to_copy": [
            {
                "old_Lev_name": "Lev4",
                "new_Lev_name": "Lev4",
                "using_constant_AMR_tol": True,
                "this_lev_is_continuation": False,
                "new_Lev_for_AMR_tolerance": 4,
                "new_Lev_for_GrDomain_and_AmrDriver": 4,
                "AH_factor": 1000.0,
                # "segments_to_copy": ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL'],
                "segments_to_copy": [
                    "AA",
                    "AB",
                    "AC",
                    "AD",
                    "AE",
                    "AF",
                    "AG",
                    "AH",
                    "AI",
                    "AJ",
                ],
                "files_to_change_in_the_new_lev": {
                    "AmrTolerances.input": {
                        "original_str": [r"ODETolerance=([^;]*);"],
                        "replaced_str": [
                            f"ODETolerance = {0.000216536 / 2000 * 4 ** (-4)};"
                        ],
                    },
                    "Evolution.input": {
                        "original_str": [
                            r"FinalTime = \d*;",
                        ],
                        "replaced_str": [
                            "FinalTime = 35000;",
                        ],
                    },
                },
            },
        ],
    },
}


verify_data_dict(data_dict)

# for key in data_dict:
#   create_new_folder(key,data_dict)
