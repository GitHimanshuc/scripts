# Copies the checkpoint folder give as input in the current directory and does ApplyObsever

called_dir=$(pwd)

cp_folder_input="$1"

# bin folder of the particular run
bin_folder=$(cd $cp_folder_input/../../../bin && pwd)

# parent directory of the script, input files will be read from there
parent_dir=$(dirname "$(dirname "$(realpath "$0")")")

# Goto the checkpoint directory
cd $cp_folder_input &&\

cp_dir_name=${PWD##*/} &&\
cp_folder=$(pwd) &&\

#copy the checkpoint files to the called directory
cp -r -v $cp_folder $called_dir &&\

#copy of the checkpoint files in the called dir
cp_called="$called_dir/$cp_dir_name" &&\

#copy all input and txt files from the parent of the checkpoint folder
cp -v ../../*.input $cp_called || : &&\
cp -v ../../*.txt $cp_called || : &&\

cp -v ../*.input $cp_called || : &&\
cp -v ../*.txt $cp_called || : &&\

cd $cp_called &&\
if test -f "./GrDomain.input"; then
    echo "GrDomain.input detected. It is a BBH simulation, using Gaugecheck_small_bbh"
    cp $parent_dir/observe_from_checkpoint_files/helper_files/Gaugecheck_small_bbh.input .
    # save the apply observer command
    command="$bin_folder/ApplyObservers -t psi,kappa -r 11,122 -d 4,4 -domaininput "./GrDomain.input" -h5prefix 'Cp-VarsGr' ./Gaugecheck_small_bbh.input"
    echo "$command" > ./run_apply_observer.sh

else

    echo "Domain.input detected. It is a single BH simulation, using Gaugecheck_small_bh"
    # echo "Adding HistoryFile=<<NONE>>; to Domain.input"
    # echo "HistoryFile=<<NONE>>;" >> ./Domain.input
    cp $parent_dir/observe_from_checkpoint_files/helper_files/Gaugecheck_small_bh.input .
    # call the apply observer
    command="$bin_folder/ApplyObservers -t psi,kappa -r 11,122 -d 4,4 -domaininput "./Domain.input" -h5prefix 'Cp-Vars' ./Gaugecheck_small_bh.input"
    echo "$command" > ./run_apply_observer.sh

fi
# Remove all domains but the spheres surrounding the BHs
echo "sed '/Cylinder/d' GaugeVis.pvd | sed '/SphereC/d'> just_BHs.pvd" >> ./run_apply_observer.sh

touch ./data_location.txt &&\
echo $cp_folder >> ./data_location.txt &&\



var_name="${cp_folder////_}" &&\
cd .. &&\
mv $cp_called $var_name