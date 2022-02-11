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


