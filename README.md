## To install the unsupported version of gcc
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt-get update
$ sudo apt-get install gcc g++ gcc-5 g++-5 gcc-6 g++-6 gcc-10 g++-7 gfortran-10

~There is also an option to change the version which will be used on calling gcc, but I do not think that it is a good idea to do that~

#Build systems
### Some make file commands for debugging
--justprint (-n), --print-data-base (-p), and --warn-undefined-variables.

#SXS

Remove all domains but the spheres surrounding the BHs
```
sed '/Cylinder/d' GaugeVis.pvd | sed '/SphereC/d'> just_BHs.pvd
```

# Docker
To use mpi inside a docker container. ~Do not use this anywhere else as it can brick your system~
```
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```


```
docker run -it --name dealii_dev -v $(pwd):/home/ dealii/dealii
```

### Dealii
(Usually the version of dealii in the docker and in the master branch is different so do a git checkout <docker branch> to run the examples)

#### Remove apt cache to make the image smaller
RUN rm -rf /var/lib/apt/lists/*


# Git

## Tags
```bash
git fetch --tags
git checkout tags/<tag_name>
git checkout tags/<tag_name> -b <branch_name> # If you want to work on the tag instead of just exploring
```


## Git shallow pull
```bash
git clone --depth=1 <repo>
git fectch --depth=100 
git pull --unshallow
```

## Git hook magic
Save the file pre_commit.sample as pre_commit and __make it executable__ in .git/hooks.

Any code added to it will be executed while commiting.
Also, get out of the .git directory before using any git commands.
```bash
conda activate working
cd /home/himanshu/Desktop/master_project/thesis/codes
jupyter nbconvert ./*.ipynb --to="python"
git add ./*
```


### Better way of doing this using .gitattributes + global .gitignore

add to .gitconfig
```
[core]
    excludesfile = ~/.gitignore
[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

now anything added to ~/.gitignore will be not be tracked in any of your local repos


add a .gitattributes file to any repo containing python notebooks with the following:
```
*.ipynb filter=strip-notebook-output
```


# GCP
## File transfer
### Copy to local system from compute instance 
```bash
gcloud compute scp --recurse [instance_name]:[folder_path_in_gcp] [local_dir]
```
### Copy to a bucket from a compute instance 
```bash
gsutil cp -r gauss gs://[bucket_name]
```


# Bash
#### List all files in the subfolder
```bash
find . -name "*.bak" -type f
```

#### Delete all files in the subfolder
```bash
find . -name "*.bak" -type f -delete
```

# Paraview

```pvserver -sp=11123``` # on the remote

then ssh into the remote with: ```-L 11111:localhost:11123```

Finally add a new server in paraview and set the port to 11123.

### vtk U(x,y); U as a surface on xy plane
extract surface1

wrap by scalar then select U

### csv (x,y,U); U as a surface on xy plane
table to points

Delaunay2d


# Compiler explorer

```bash
docker run -it -v $(pwd):/opt/compiler-explorer -p 10240:10240 ubuntu:22.04

apt update
apt upgrade -y
apt-get install -y ca-certificates curl gnupg zsh git make

# install node
mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
NODE_MAJOR=20
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
apt update
apt install nodejs -y

# install compiler explorer
cd
git clone https://github.com/compiler-explorer/compiler-explorer.git --depth=1
cd compiler-explorer
make dev

# install oh my zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# install infra
cd 
git clone https://github.com/compiler-explorer/infra.git --depth=1
cd infra
make ce
./bin/ce_install list
```


# VS code debug config for spec
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "/workspaces/spec/Tests/BlackBoxTests/GeneralizedHarmonicExamples/SingleBH_DhGauge/Run/Lev3/SpEC",
            "args": [
                "> SpEC.out 2>&1"
            ],
            "stopAtEntry": true,
            "cwd": "/workspaces/spec/Tests/BlackBoxTests/GeneralizedHarmonicExamples/SingleBH_DhGauge/Run/Lev3/",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Set Disassembly Flavor to Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ]
        },
        {
            "name": "BBH_ID.py",
            "type": "python",
            "python": "/root/miniconda3/envs/working/bin/python",
            "request": "launch",
            "program": "/workspaces/spec/InitialValueProblem/ScriptsIVP/BBH_ID.py",
            "args": [
                "--q=3",
                "--chiA=0,0,0",
                "--chiB=0,0,0",
                "--D=10",
                "--Omega0=0.0279722304683467",
                "--adot=-0.000202425618780214",
                "--levels=6",
                "--nprocs=5",
                "--its=100",
                "--IDType=SKS"
            ],
            "cwd": "/workspaces/spec/not_tracked/ID_test/ID_1/ID",
            "justMyCode": true
        }
    ]
}
```
