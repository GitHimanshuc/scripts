import numpy as np
import pandas as pd
import sys


# Example dat file
"""
# ProportionalIntegral 
# [1] = time after step
# [2] = NumRhsEvaluations in this segment
# [3] = NgoodSteps+NfailedSteps
# [4] = NgoodSteps
# [5] = NfailedSteps
# [6] = error/1e-08
# [7] = dt
# [8] = dt attempted
# [9] = dt desired by control
# [10] = courant factor
  1.0000000000000000e-03     7     1     1    0 1.87842e-08  0.00100000  0.00100000  0.00100000   0.0709092
.
.
.
.

"""

# Usage
"""
python dat_to_csv.py <dat_file_input> <(optional)output_file>
"""

# .dat files have a weird arrangement there can be a comment on top, follwed by column names per line and then the data. This function reads the whole stuff and returns a csv file of the same name.
def read_dat_file(file_name):
  cols_names = []

  temp_file = "./temp.csv"
  with open(file_name, 'r') as f:
    with open(temp_file, 'w') as w:
      lines = f.readlines()
      for line in lines:
        if(line[0] != '#'):  # This is data
          w.writelines(line)
        if(line[0:3] == '# ['):  # Some dat files have comments on the top
          cols_names.append(line.split('=')[-1][1:-1].strip())

  return pd.read_csv(temp_file, delim_whitespace=True, names=cols_names)


data = read_dat_file(sys.argv[1])

if len(sys.argv)==2:
  data.to_csv(sys.argv[1][:-3]+"csv")
elif len(sys.argv)==3:
  data.to_csv(sys.argv[2])
else:
  print("""
  Wrong usage

  python dat_to_csv.py <dat_file_input> <(optional)output_file>
  """)

