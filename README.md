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



