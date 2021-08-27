#!/bin/bash -l
#BSUB -q batch
#BSUB -J WL-Fe_n0001
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-LSMS/WL_Fe_n0001/Run_Archive/1613596055.6940515/WL-Fe_n0001.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-LSMS/WL_Fe_n0001/Run_Archive/1613596055.6940515/WL-Fe_n0001.e%J
#BSUB -nnodes 1
#BSUB -W 60
#BSUB -P csc425

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load olcf_harness_summit
module load gcc
module load cuda
module load spectrum-mpi
module load essl
module load hdf5
module list

#-----------------------------------------------------
# Define some variables.                             -
#-----------------------------------------------------
EXECUTABLE="bin"
STARTINGDIRECTORY="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-LSMS/WL_Fe_n0001/Scripts"
WORKDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-LSMS/WL_Fe_n0001/1613596055.6940515/workdir"
RESULTSDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-LSMS/WL_Fe_n0001/Run_Archive/1613596055.6940515"
UNIQUE_ID_STRING="1613596055.6940515"
INPUT=./i_lsms_16
NUM_LSMS=10
STEPS_PER_WALKER=10
MPI_PER_LSMS=2
NUM_ATOMS=16

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
cp $STARTINGDIRECTORY/Inputs/$INPUT .
cp $STARTINGDIRECTORY/Inputs/v_fe2.0 .

#-----------------------------------------------------
# Run the executable.                                -
#-----------------------------------------------------

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

echo "Running lsms using $INPUT with $NUM_LSMS walkers..."

echo "jsrun --progress ./progress_lsms.${LSB_JOBID}.txt --exit_on_error 1 --nrs $TOTAL_MPI --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --bind rs $EXECUTABLE $INPUT -mode 1d -size_lsms ${NUM_ATOMS} -num_lsms ${NUM_LSMS} -num_steps ${TOTAL_STEPS} -energy_calculation multistep  1> stdout.txt 2> stderr.txt"
jsrun --progress ./progress_lsms.${LSB_JOBID}.txt --exit_on_error 1 --nrs $TOTAL_MPI --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --bind rs $EXECUTABLE $INPUT -mode 1d -size_lsms ${NUM_ATOMS} -num_lsms ${NUM_LSMS} -num_steps ${TOTAL_STEPS} -energy_calculation multistep  1> stdout.txt 2> stderr.txt

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
case 0 in
    0) 
       test_harness_driver.py -r;;

    1) 
       echo "No resubmit";;
esac 

