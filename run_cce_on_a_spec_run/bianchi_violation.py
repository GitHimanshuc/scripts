# Call from the cce folder and this will load the path dict from the pickel and start the bianchi stuff.
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,10)
import json
import pickle
from pathlib import Path
import scri
import scipy
spec_home="/home/hchaudha/spec"

def plot_and_save_bianchi_violations(violation_dict:dict,save_dir:Path):
    plt.semilogy(violation_dict['t'],violation_dict['5'],label='5')
    plt.semilogy(violation_dict['t'],violation_dict['4'],label='4')
    plt.semilogy(violation_dict['t'],violation_dict['3'],label='3')
    plt.semilogy(violation_dict['t'],violation_dict['2'],label='2')
    plt.semilogy(violation_dict['t'],violation_dict['1'],label='1')
    plt.semilogy(violation_dict['t'],violation_dict['0'],label='0')
    plt.xlabel('t')
    plt.ylabel("violations")
    plt.legend()
    plt.savefig(save_dir/"violations.png")

def post_process_jobs(path_dict):

    bianchi_violations={}
    for bd_data_path in path_dict['boundary_data_paths']:
        bd_folder_path = bd_data_path.parent

        directory = str(bd_folder_path)+"/"
        
        # compute bianchi violations and save the pickel

        abd = scri.create_abd_from_h5(
          file_name=str(bd_data_path),
          file_format="spectrecce_v1",
          # ch_mass=1.0,  # Optional; helpful if known
          # t_interpolate=t_worldtube,  # Optional; for some specified values of `t_worldtube`
          # t_0_superrest=1240,
          # padding_time=2
      )
      
        violations = abd.bondi_violation_norms

        # dump the dict as a pickel
        violations_dict = {
          't': abd.t,
          '0': violations[0],
          '1': violations[1],
          '2': violations[2],
          '3': violations[3],
          '4': violations[4],
          '5': violations[5]
        }
        with open(f'{directory}bondi_violation_dict.pkl','wb') as f:
            pickle.dump(violations_dict,f)
        plot_and_save_bianchi_violations(violations_dict,bd_folder_path)

        total_violations = []
        for violation in violations:
            total_violations.append(scipy.integrate.trapezoid(violation, abd.t))
            bianchi_violations[str(bd_folder_path.stem)] = total_violations


    # Save the bianchi violation dict
    with open(f'{directory}bianchi_violations.json', 'w') as f:
        json.dump(bianchi_violations, f, indent=2, separators=(",", ": "), ensure_ascii=True)

path_dict = {}
with open("./path_dict.pkl",'rb') as f:
    path_dict = pickle.load(f)

post_process_jobs(path_dict)