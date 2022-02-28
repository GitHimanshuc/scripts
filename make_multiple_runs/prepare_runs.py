import subprocess
import sys
import json
import os
import shutil

def generate_params_file(mass_ratio=1, spinA=(0, 0, 0), spinB=(0, 0, 0), D0=10):

  command = f"/home/hchaudha/spec/Support/bin/ZeroEccParamsFromPN --q \"{mass_ratio}\" --chiA \"{spinA[0]},{spinA[1]},{spinA[2]}\" --chiB \"{spinB[0]},{spinB[1]},{spinB[2]}\" --D0 \"{D0}\""

  # Generate a temporary shell file to run the script because directly calling
  # the command is not working
  shell_file = "./temp_params_generation_file.sh"
  with open(shell_file, 'w') as file:
    file.write(command)

  # Save the output
  data = subprocess.run(["zsh", shell_file], capture_output=True)
  # print(data.stdout)

  # Delete the temp file
  subprocess.run(f"rm ./{shell_file}".split())

  parameters = data.stdout.splitlines()[-13:]
  parameters = [str(i).replace('\'', '') for i in parameters]

  # Params.input
  param_file = f"""# Set the initial data parameters

  # Orbital parameters
  $Omega0 = {parameters[1].split("= ")[1]};
  $adot0 = {parameters[3].split("= ")[1]};
  $D0 = {D0};

  # Physical parameters (spins are dimensionless)
  $MassRatio = {mass_ratio};
  @SpinA = {spinA};
  @SpinB = {spinB};

  # Evolve after initial data completes?
  $Evolve = 1;

  # IDType: "SKS", "SHK", "SSphKS" or "CFMS".
  $IDType = "SKS";

  # Expected Norbits: {parameters[-2].split("= ")[-1]}
  # Expected tMerger: {parameters[-1].split("= ")[-1]}

  """

  # Write the generated params file
  with open("./Params.input", 'w') as f:
    f.write(param_file)
    f.write("# ",command)



def checkout_and_compile_branch(branch_name):
  subprocess.run(f"zsh ./checkout_and_compile.sh {branch_name}".split())

def prepare_ID(folder_path):
  subprocess.run(f"zsh ./call_prepare_id.sh {folder_path}".split())

def submit_job(folder_path):
  subprocess.run(f"zsh ./submit.sh {folder_path}".split())


def read_data_from_json_file(file_location):
  data=""
  with open (file_location, "r") as f:
    data = json.loads(f.read())

  return data["params"]


def create_simulation_folders(file_location="./runs_data.json"):
  runs_data = read_data_from_json_file(file_location)

  for data in runs_data:
    git_branch = data["git_branch"]
    checkout_and_compile_branch(git_branch)

    mass_ratio = data["mass_ratio"]
    spinA = tuple(data["spinA"])
    spinB = tuple(data["spinB"])
    D0 = data["D0"]

    generate_params_file(mass_ratio,spinA,spinB,D0)

    dir_name = "./" + git_branch.replace("/","_") + f"_{mass_ratio}_{spinA[0]}_{spinA[1]}_{spinA[2]}_{spinB[0]}_{spinB[1]}_{spinB[2]}_{D0}"

    os.makedirs(dir_name)
    prepare_ID(dir_name)

    shutil.copy("./Params.input",f"{dir_name}/Params.input")
    shutil.copy("./DoMultipleRuns.input",f"{dir_name}/Ev/DoMultipleRuns.input")

    submit_job(dir_name)
    print("DONE: ",dir_name)


if __name__ == '__main__':
  # pass the 
  create_simulation_folders(sys.argv[1])