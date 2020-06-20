## To install the unsupported version of gcc
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt-get update
$ sudo apt-get install gcc g++ gcc-5 g++-5 gcc-6 g++-6 gcc-10 g++-7 gfortran-10

~There is also an option to change the version which will be used on calling gcc, but I do not think that it is a good idea to do that~
#Build systems
### Some make file commands for debugging
--justprint (-n), --print-data-base (-p), and --warn-undefined-variables.

# Docker
docker run -it -v $(pwd):/home/ dealii/dealii
#### Remove apt cache to make the image smaller
RUN rm -rf /var/lib/apt/lists/*


# Git
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


