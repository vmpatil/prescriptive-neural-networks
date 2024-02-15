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
mkdir warfarin && tar -xzvf warfarin.tar.gz -C warfarin

# run python script
python3 cf.py

# tar zip the output directory

tar -czvf output.tar.gz output
