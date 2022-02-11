called_dir=$(pwd)

cp_folder_input="$1"

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
    # call the apply observer
    ApplyObservers -t psi,kappa -r 11,122 -d 4,4 -domaininput "./GrDomain.input" -h5prefix 'Cp-VarsGr' $scripts_path/helper_files/Gaugecheck_small_bbh.input

else

    echo "Domain.input detected. It is a single BH simulation, using Gaugecheck_small_bh"
    # echo "Adding HistoryFile=<<NONE>>; to Domain.input"
    # echo "HistoryFile=<<NONE>>;" >> ./Domain.input
    # call the apply observer
    ApplyObservers -t psi,kappa -r 11,122 -d 4,4 -domaininput "./Domain.input" -h5prefix 'Cp-Vars' $scripts_path/helper_files/Gaugecheck_small_bh.input

fi
# Remove all domains but the spheres surrounding the BHs
sed '/Cylinder/d' GaugeVis.pvd | sed '/SphereC/d'> just_BHs.pvd &&\

touch ./data_location.txt &&\
echo $cp_folder >> ./data_location.txt &&\



var_name="${cp_folder////_}" &&\
cd .. &&\
mv $cp_called $var_name