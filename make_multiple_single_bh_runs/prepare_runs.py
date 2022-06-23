import subprocess
import sys
import json
import os
import shutil
import re
from datetime import datetime

if "central" in os.getcwd(): # We are on caltech HPC
  spec_home = "/home/hchaudha/spec"
elif "panfs" in os.getcwd(): # Wheeler
  spec_home = "/home/himanshu/spec/my_spec"
else:
  print("Machine not recognized\n")
  exit(0)

def checkout_and_compile_branch(branch_name):
  subprocess.run(f"zsh ./checkout_and_compile.sh {spec_home} {branch_name}".split())

def prepare_EV(folder_path):
  subprocess.run(f"zsh ./call_prepare_ev.sh {folder_path}".split())

def submit_job(folder_path):
  subprocess.run(f"zsh ./submit.sh {folder_path}".split())


def read_data_from_json_file(file_location):
  data=""
  with open (file_location, "r") as f:
    data = json.loads(f.read())

  return data["params"]


def add_to_start_jobs_script(dir):
  with open("./start_jobs.sh",'a') as file:
    file.write(f"cd {dir} && ./StartJob.sh\n")

def replace_current_file(file_path,original_str,replaced_str):
    with open(file_path,'r') as file:
        data = file.read()
        
    data,replaced_status = re.subn(original_str,replaced_str,data)
    if replaced_status != 0:
        print(f"""
        Replaced in File: {file_path}
        Original String: {original_str}
        Replaced String: {replaced_str}
        """)
    else:
        print(f"""
        !!!!FAILED TO REPLACE!!!!
        File path: {file_path}
        Original String: {original_str}
        Replaced String: {replaced_str}
        """)
    
    with open(file_path,'w') as file:
        file.write(data)
        
        
def replace_files(curr_run,run_dir):
    if 'file_replace' in curr_run:
        for i in curr_run['file_replace']:
            if(len(i['original_str']) != len(i['replaced_str'])):
                print("original_str and replaced_str should have the same length.")
                sys.exit("original_str and replaced_str should have the same length.")
            else:
                for original_str,replaced_str in zip(i['original_str'],i['replaced_str']):
                    replace_current_file(run_dir+i["file_path"],original_str,replaced_str)


def create_simulation_folders(file_location="./runs_data.json",dry_run=False):
  runs_data = read_data_from_json_file(file_location)

  for data in runs_data:
    git_branch = data["git_branch"]
    if(dry_run==False):
      checkout_and_compile_branch(git_branch)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "folder_name" not in data:
      dir_name = "./" + git_branch.replace("/","_") + f"_{current_datetime}"
    else:
      dir_name = "./" + data["folder_name"]
    os.makedirs(dir_name)

    prepare_EV(dir_name)

    replace_files(data,dir_name)

    if(dry_run==False):
      submit_job(dir_name)
      
    add_to_start_jobs_script(dir_name)
    print(f"""
    DONE: {dir_name}
    ###########################################################################
    ###########################################################################

    """)

if __name__ == '__main__':
  if len(sys.argv) == 2 :
    create_simulation_folders(sys.argv[1])
  elif sys.argv[2] == "dry":
    create_simulation_folders(sys.argv[1],True)
  else:
    print("prepare_runs.py ./data.json dry")