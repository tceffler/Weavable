#! /bin/bash -l
#BSUB -P __projectid__
#BSUB -q __batchqueue__
#BSUB -J __jobname__
#BSUB -o __resultsdir__/__jobname__.o.%J
#BSUB -e __resultsdir__/__jobname__.e.%J
#BSUB -nnodes __nnodes__
#BSUB -alloc_flags "smt1 cpublink"
#BSUB -W __walltime__

#-----------------------------------------------------
# Function name: launch_a_copy_with_helper_scripts   -
#                                                    -
# Arguments:                                         -
#   ${1} The id of this snap copy.                   -
#   ${2} The working dir of the test                 -
#   ${3} The dirname of the base job                 -
#        input files.                                -
#   ${4} The path to the snap binary.                -
#                                                    -
# Description:                                       -
#   A subdirectory is made within the working        -
#   directory. All input files  are copied to the    -
#   subdirectory, and the the snap instance is       -
#   launched from this subdirectory.                 -
#                                                    -
#   The jsrun command is run in the background.      -
#                                                    -
#   The IBM helper script is used to control         -
#   job placement.                                   -
#                                                    -
#-----------------------------------------------------
function launch_a_copy_with_helper_scripts() {
    declare -r copy_no=${1} 
    declare -r work_dir=${2}
    declare -r dirname_of_inputfiles=${3}
    declare -r snap_binary=${4}

    declare -r starting_dir=$(pwd)
    declare -r exe_dir=${work_dir}/snap_copy_${1}
    tstamp=`date +%m_%d_%H_%M_%S`

    # Make the execution directory for the copy of the launch.
    mkdir -p ${exe_dir}

    # Copy all inputfiles to the execution directory
    # for this copy of snap.
    cp ${dirname_of_inputfiles}/input.96x84_516x480x504 ${exe_dir}
    cp ${dirname_of_inputfiles}/h6.sh ${exe_dir}
    cp ${dirname_of_inputfiles}/numactl ${exe_dir}

    # Launch this instance of snap.
    cd ${exe_dir}
    jsrun -X 1 \
    --progress jsrun.progress.${copy_no}.log \
    --nrs 192  --tasks_per_rs 42 --cpu_per_rs 42 \
    -d plane:42 --bind rs \
    ./h6.sh  ${snap_binary} ./input.96x84_516x480x504 out.cpu.$tstamp 1> snap.stdout.txt 2> snap.stderr.txt  &

    # Change back to the starting directory.
    cd ${starting_dir}
}

#-----------------------------------------------------
# Function name: launch_a_copy_no_helper_scripts     -
#                                                    -
# Arguments:                                         -
#   ${1} The id of this snap copy.                   -
#   ${2} The working dir of the test                 -
#   ${3} The dirname of the base job                 -
#        input files.                                -
#   ${4} The path to the snap binary.                -
#                                                    -
# Description:                                       -
#   A subdirectory is made within the working        -
#   directory. All input files  are copied to the    -
#   subdirectory, and the the snap instance is       -
#   launched from this subdirectory.                 -
#                                                    -
#   The jsrun command is run in the background.      -
#                                                    -
#-----------------------------------------------------
function launch_a_copy_no_helper_scripts() {
    declare -r copy_no=${1} 
    declare -r work_dir=${2}
    declare -r dirname_of_inputfiles=${3}
    declare -r snap_binary=${4}

    declare -r starting_dir=$(pwd)
    declare -r exe_dir=${work_dir}/snap_copy_${1}
    tstamp=`date +%m_%d_%H_%M_%S`

    # Make the execution directory for the copy of the launch.
    mkdir -p ${exe_dir}

    # Copy all inputfiles to the execution directory
    # for this copy of snap.
    cp ${dirname_of_inputfiles}/input.96x84_516x480x504 ${exe_dir}
    cp ${dirname_of_inputfiles}/h6.sh ${exe_dir}
    cp ${dirname_of_inputfiles}/numactl ${exe_dir}

    # Launch this instance of snap.
    cd ${exe_dir}
    jsrun -X 1 \
    --nrs 192  --tasks_per_rs 42 --cpu_per_rs 42 \
    -d plane:42 --bind rs \
    ${snap_binary} ./input.96x84_516x480x504  out.cpu.$tstamp 1> snap.stdout.txt 2> snap.stderr.txt &

    # Change back to the starting directory.
    cd ${starting_dir}
}

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#                                                    -
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load __nccstestharnessmodule__

#-----------------------------------------------------
# Define some variables.                             -
#                                                    -
#-----------------------------------------------------
EXECUTABLE="__pathtoexecutable__"
STARTINGDIRECTORY="__startingdirectory__"
WORKDIR="__workdir__"
RESULTSDIR="__resultsdir__"
UNIQUE_ID_STRING="__uniqueidstring__"
DIRNAME_OF_INPUTFILES="__dirnameinputfiles__"

export OMP_NUM_THREADS=1
export BIND_SLOTS=4
export RANKS_PER_NODE=42
export BIND_THREADS=yes
export SYSTEM_CORES=2
export DISABLE_ASLR=yes
export OMPI_MCA_pml_pami_local_eager_limit=16001
export OMPI_LD_PRELOAD_POSTPEND=/ccs/home/walkup/mpitrace/spectrum_mpi/libmpihpm.so 

#-----------------------------------------------------
# Enusre that we are in the correct starting         -
# directory.                                         -
#                                                    -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Make the working scratch space directory.          -
#                                                    -
#-----------------------------------------------------
if [ ! -e $WORKDIR ]
then
    mkdir -p $WORKDIR
fi

#-----------------------------------------------------
# Make the results directory.                        -
#                                                    -
#-----------------------------------------------------
if [ ! -e $RESULTSDIR ]
then
    mkdir -p $RESULTSDIR
fi

#-----------------------------------------------------
#  Change directory to the working directory.        -
#                                                    -
#-----------------------------------------------------
cd $WORKDIR

#-----------------------------------------------------
# Run the executable.                                -
#                                                    -
#-----------------------------------------------------
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

for (( counter=0; counter<__maxcopies__; counter+=1 ))
do
   launch_a_copy_with_helper_scripts ${counter} ${WORKDIR} ${DIRNAME_OF_INPUTFILES} ${EXECUTABLE}
done

#-----------------------------------------------------
# The jsrun launch command is running in the         -
# background. We need to wait for the command to     -
# complete before we can copy the results.           -
#                                                    -
#-----------------------------------------------------
sleep 30

jslist

wait

jslist

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

#-----------------------------------------------------
# Enusre that we return to the starting directory.   -
#                                                    -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Move the batch file name to  $RESULTSDIR           -
#                                                    -
#-----------------------------------------------------
mv __batchfilename__ $RESULTSDIR

#-----------------------------------------------------
# Check the final results.                           -
#                                                    -
#-----------------------------------------------------
# check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

#-----------------------------------------------------
# The script now determines if we are to resubmit    -
# itself.                                            -
#                                                    -
#-----------------------------------------------------
case __resubmitme__ in
    0) 
       test_harness_driver.py -r;;

    1) 
       echo "No resubmit";;
esac 


#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR           -
#                                                    -
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR

