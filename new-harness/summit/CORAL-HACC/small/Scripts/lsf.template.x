#! /bin/bash -l
#BSUB -q __batch_queue__
#BSUB -J __job_name__
#BSUB -o __results_dir__/__job_name__.o%J
#BSUB -e __results_dir__/__job_name__.e%J
#BSUB -W __walltime__
#BSUB -P CSC425
#BSUB -nnodes __nodes__
#BSUB -alloc_flags "smt1 cpublink"

#-----------------------------------------------------
# Set up the environment for use of the harness
#-----------------------------------------------------
# Load the modules and enviornment
module unload xl gcc essl cuda spectrum-mpi
module load gcc
module load essl
module load cuda
module load spectrum-mpi
module list

#-----------------------------------------------------
# Define some variables
#-----------------------------------------------------
EXECUTABLE="./build_directory/__executable_path__"
STARTINGDIRECTORY="__scripts_dir__"
WORKDIR="__working_dir__"
RESULTSDIR="__results_dir__"
#UNIQUE_ID_STRING="__unique_id_string__"
UNIQUE_ID_STRING="__harness_id__"
INPUTDIR=${STARTINGDIRECTORY}/../Inputs
EXE_FULL=`readlink -f $EXECUTABLE`
#-----------------------------------------------------
# Ensure that we are in the correct starting directory
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Make the working scratch space directory
#-----------------------------------------------------
mkdir -p $WORKDIR
sleep 10

#-----------------------------------------------------
# Make the results directory
#-----------------------------------------------------
mkdir -p $RESULTSDIR
sleep 10

#-----------------------------------------------------
#  Change directory to the working directory
#-----------------------------------------------------
cd $WORKDIR

echo "Changed to working directory"
cp ${INPUTDIR}/* .
pwd
ls -l


#-----------------------------------------------------
# Set enviornmental variables
#-----------------------------------------------------
ulimit -c 0
export OMP_NUM_THREADS=1
export OMP_TARGET_OFFLOAD=MANDATORY


#-----------------------------------------------------
# Run the executable
#-----------------------------------------------------
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

JSRUN_OPTIONS="-X 1 --progress ${tstamp}.progress --nrs 8 --tasks_per_rs 1 --cpu_per_rs 8 --gpu_per_rs 1 --rs_per_host 4 --bind=none -l GPU-CPU --stdio_mode collected --stdio_stdout stdout.txt --stdio_stderr stderr.txt"

echo jsrun --smpiargs="-mca coll ^ibm" ${JSRUN_OPTIONS} $EXE_FULL ./indat ./cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 512 -t 2x2x2
time jsrun --smpiargs="-mca coll ^ibm" ${JSRUN_OPTIONS} $EXE_FULL ./indat ./cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 512 -t 2x2x2

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final
sleep 30


#-----------------------------------------------------
# Enusre that we return to the starting directory.
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR && rm -rf $WORKDIR

#-----------------------------------------------------
# Move the batch file name to  $RESULTSDIR
#-----------------------------------------------------
mv __batchfilename__ $RESULTSDIR

#-----------------------------------------------------
# Check the final results
#-----------------------------------------------------
check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

#-----------------------------------------------------
# The script now determines if we are to resubmit itself                                           -
#-----------------------------------------------------
case __resubmitme__ in
    0) 
       test_harness_driver.py -r;;
    1) 
       echo "No resubmit";;
esac 
