#!/bin/bash -l
#BSUB -q __batch_queue__
#BSUB -J __job_name__
#BSUB -o __results_dir__/__job_name__.o%J
#BSUB -e __results_dir__/__job_name__.e%J
#BSUB -nnodes __nodes__
#BSUB -W __walltime__
#BSUB -P __project_id__

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load __nccstestharnessmodule__
module load gcc
module load cuda
module load spectrum-mpi
module load essl
module load hdf5
module list

#-----------------------------------------------------
# Define some variables.                             -
#-----------------------------------------------------
EXECUTABLE="__build_dir__/__pathtoexecutable__/__executablename__"
STARTINGDIRECTORY="__scripts_dir__"
WORKDIR="__working_dir__"
RESULTSDIR="__results_dir__"
UNIQUE_ID_STRING="__harness_id__"
INPUT=__inputfile__
NUM_LSMS=__num_walkers__
STEPS_PER_WALKER=__steps_per_walker__
MPI_PER_LSMS=__mpi_per_walker__
NUM_ATOMS=__num_atoms__

TOTAL_STEPS=$(( $NUM_LSMS * $STEPS_PER_WALKER ))
TOTAL_MPI=$(( $NUM_LSMS * $MPI_PER_LSMS + 1 ))

#-----------------------------------------------------
# Ensure that we are in the correct starting         -
# directory.                                         -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Make the working scratch space directory.          -
#-----------------------------------------------------
if [ ! -e $WORKDIR ]
then
    mkdir -p $WORKDIR
fi

#-----------------------------------------------------
# Make the results directory.                        -
#-----------------------------------------------------
if [ ! -e $RESULTSDIR ]
then
    mkdir -p $RESULTSDIR
fi

#-----------------------------------------------------
#  Change directory to the working directory.        -
#-----------------------------------------------------
cd $WORKDIR

#-----------------------------------------------------
#  Copy input files to the working directory.        -
#-----------------------------------------------------
cp $STARTINGDIRECTORY/Inputs/$INPUT i_lsms
cp $STARTINGDIRECTORY/Inputs/v_fe2.* .

#-----------------------------------------------------
# Run the executable.                                -
#-----------------------------------------------------

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

echo "Running lsms using $INPUT with $NUM_LSMS walkers..."

echo "jsrun --progress ./progress_lsms.${LSB_JOBID}.txt --exit_on_error 1 --nrs $TOTAL_MPI --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --bind rs $EXECUTABLE $INPUT -mode 1d -size_lsms ${NUM_ATOMS} -num_lsms ${NUM_LSMS} -num_steps ${TOTAL_STEPS} -energy_calculation multistep  1> stdout.txt 2> stderr.txt"
jsrun --progress ./progress_lsms.${LSB_JOBID}.txt --exit_on_error 1 --nrs $TOTAL_MPI --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --bind rs $EXECUTABLE i_lsms -mode 1d -size_lsms ${NUM_ATOMS} -num_lsms ${NUM_LSMS} -num_steps ${TOTAL_STEPS} -energy_calculation multistep  1> stdout.txt 2> stderr.txt

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

#-----------------------------------------------------
# Enusre that we return to the starting directory.   -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR           -
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR && rm -rf $WORKDIR

#-----------------------------------------------------
# Move the batch file name to  $RESULTSDIR           -
#-----------------------------------------------------
mv __batch_file_name__ $RESULTSDIR

#-----------------------------------------------------
# Check the final results.                           -
#-----------------------------------------------------
check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

#-----------------------------------------------------
# The script now determines if we are to resubmit    -
# itself.                                            -
#-----------------------------------------------------
case __resubmitme__ in
    0) 
       test_harness_driver.py -r;;

    1) 
       echo "No resubmit";;
esac 

