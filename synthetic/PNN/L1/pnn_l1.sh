#!/bin/bash

# cp /mnt/ws/home/vpatil/test-env.tar.gz ./

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=test-env
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

# telling gurobi where to find the license
GUROBI_HOME="/mnt/dv/wid/projects1/WID-Software/opt-progs/gurobi810/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE="/progs/gurobi/gurobi.lic"

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# modify this line to run your desired Python script and any other work you need to do

# unzip the tar zipped data file :--> 
# mkdir 1e_athey_100 && tar -xzvf 1e_athey_100.tar.gz -C 1e_athey_100 --strip-components 1
# mkdir 1e_athey_500 && tar -xzvf 1e_athey_500.tar.gz -C 1e_athey_500 --strip-components 1
# mkdir 1e_athey_100_learned && tar -xzvf 1e_athey_100_learned.tar.gz -C 1e_athey_100_learned --strip-components 1
# mkdir 1e_athey_500_learned && tar -xzvf 1e_athey_500_learned.tar.gz -C 1e_athey_500_learned --strip-components 1

# mkdir 2e_athey_100 && tar -xzvf 2e_athey_100.tar.gz -C 2e_athey_100 --strip-components 1
# mkdir 2e_athey_500 && tar -xzvf 2e_athey_500.tar.gz -C 2e_athey_500 --strip-components 1
# mkdir 2e_athey_100_learned && tar -xzvf 2e_athey_100_learned.tar.gz -C 2e_athey_100_learned --strip-components 1
# mkdir 2e_athey_500_learned && tar -xzvf 2e_athey_500_learned.tar.gz -C 2e_athey_500_learned --strip-components 1

mkdir 3e_athey_100 && tar -xzvf 3e_athey_100.tar.gz -C 3e_athey_100 --strip-components 1
mkdir 3e_athey_500 && tar -xzvf 3e_athey_500.tar.gz -C 3e_athey_500 --strip-components 1
mkdir 3e_athey_100_learned && tar -xzvf 3e_athey_100_learned.tar.gz -C 3e_athey_100_learned --strip-components 1
mkdir 3e_athey_500_learned && tar -xzvf 3e_athey_500_learned.tar.gz -C 3e_athey_500_learned --strip-components 1

# run python script
python3 pnn_l1.py

# tar zip the output directory
# tar -czvf 1_easy.tar.gz 1_Easy
# tar -czvf 2_easy.tar.gz 2_Easy
tar -czvf 3_easy.tar.gz 3_Easy
