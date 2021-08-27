#! /bin/bash -l
#BSUB -q __batch_queue__
#BSUB -J __job_name__
#BSUB -o __results_dir__/__job_name__.o%J
#BSUB -e __results_dir__/__job_name__.e%J
#BSUB -W __walltime__
#BSUB -P CSC425
#BSUB -nnodes __nodes__
#BSUB -alloc_flags "gpumps"

#-----------------------------------------------------
# Set up the environment for use of the harness
#-----------------------------------------------------
#source __rgtenvironmentalfile__
#module load __nccs_test_harness_module__
#module unload xalt
#module load __compiler_modulefile__
#module load cuda/9.2.148
#module load spectrum-mpi
module purge

# Load modules
module load gcc
module load cmake
module load cuda/9.2.148
module load spectrum-mpi
OLCF_HARNESS_DIR=/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/new-harness/olcf-test-harness
module use $OLCF_HARNESS_DIR/modulefiles
module load olcf_harness
module list

#-----------------------------------------------------
# Define some variables
#-----------------------------------------------------
EXECUTABLE="__executable_path__"
STARTINGDIRECTORY="__scripts_dir__"
WORKDIR="__working_dir__"
RESULTSDIR="__results_dir__"
#UNIQUE_ID_STRING="__unique_id_string__"
UNIQUE_ID_STRING="__harness_id__"
#INPUTDIR=${STARTINGDIRECTORY}/../Inputs

#-----------------------------------------------------
# Get the number of resource sets
#-----------------------------------------------------
if [ "__nodes__" -eq "188" ]
then
  nrs=1125
elif [ "__nodes__" -eq "96" ]
then
  nrs=576
elif [ "__nodes__" -eq "12" ]
then
  nrs=72
elif [ "__nodes__" -eq "2" ]
then
  nrs=9
else
  echo "Invalid node count"
  exit
fi

#-----------------------------------------------------
# Enusre that we are in the correct starting directory
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Make the working scratch space directory
#-----------------------------------------------------
if [ ! -e $WORKDIR ]
then
    mkdir -p $WORKDIR
fi

#-----------------------------------------------------
# Make the results directory
#-----------------------------------------------------
if [ ! -e $RESULTSDIR ]
then
    mkdir -p $RESULTSDIR
fi

#-----------------------------------------------------
#  Change directory to the working directory
#-----------------------------------------------------
cd $WORKDIR

echo "Changed to working directory"
pwd
ls -l

#-----------------------------------------------------
# Run the executable
#-----------------------------------------------------
#HELPER=/ccs/home/hfwen/bin/jsm_setup.sh
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

ulimit -s 10240
unset OMP_NUM_THREADS

JSRUN_OPTIONS="--nrs ${nrs} --tasks_per_rs 3 --cpu_per_rs 6 --gpu_per_rs 1 -b packed:2 -d plane:3 --stdio_mode collected --stdio_stdout stdout.txt --stdio_stderr stderr.txt"

echo jsrun ${JSRUN_OPTIONS} ${HELPER} ${EXECUTABLE} -s 128 -i 5000
jsrun ${JSRUN_OPTIONS} ${HELPER} __build_dir__/bin/${EXECUTABLE} -s 128 -i 5000

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

sleep 30

#-----------------------------------------------------
# Ensure that we return to the starting directory
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR #&& rm -rf $WORKDIR

#-----------------------------------------------------
# Move the batch file name to  $RESULTSDIR
#-----------------------------------------------------
mv __batch_filename__ $RESULTSDIR

#-----------------------------------------------------
# Check the final results
#-----------------------------------------------------
check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

#-----------------------------------------------------
# The script now determines if we are to resubmit itself.                                            -
#-----------------------------------------------------
case __resubmitme__ in
    0) 
       test_harness_driver.py -r;;
    1) 
       echo "No resubmit";;
esac 
