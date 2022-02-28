import subprocess
import sys


mass_ratio = 1
spinA = (0,0,0)
spinB = (0,0,0)
D0 = 10


if len(sys.argv) > 1:
  mass_ratio = float(sys.argv[1])
  spinA = (float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]))
  spinA = (float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]))
  D0 = float(sys.argv[8])

print(f"""
#######################################################################
#######################################################################

Generating for parameters:

mass_ratio = {mass_ratio}
spinA = {spinA}
spinB = {spinB}
D0 = {D0}

""")

command = f"/home/hchaudha/spec/Support/bin/ZeroEccParamsFromPN --q \"{mass_ratio}\" --chiA \"{spinA[0]},{spinA[1]},{spinA[2]}\" --chiB \"{spinB[0]},{spinB[1]},{spinB[2]}\" --D0 \"{D0}\""

# Generate a temporary shell file to run the script because directly calling
# the command is not working
shell_file = "./temp_params_generation_file.sh"
with open(shell_file,'w') as file:
  file.write(command)

# Save the output
data = subprocess.run(["zsh",shell_file],capture_output=True)
# print(data.stdout)


# Delete the temp file
subprocess.run(f"rm ./{shell_file}".split())


parameters = data.stdout.splitlines()[-13:]
parameters = [str(i).replace('\'','') for i in parameters]

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
  f.write(command)
