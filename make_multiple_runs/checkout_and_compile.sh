# usage
# zsh ./checkout_and_compile.sh <SPEC_HOME> <branch name>

cd $1
git checkout $2
make parallel