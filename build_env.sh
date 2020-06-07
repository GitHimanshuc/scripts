julia_link="https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.2-linux-x86_64.tar.gz"



cd ~
apt get update -y && \
apt install build-essential git wget curl nnn zsh vim nano -qq && \
mkdir software && cd software && \
git clone https://github.com/spack/spack.git && \
curl https://getmic.ro | bash && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
wget $julia_link && \



# After this things have to be done manually 
sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
