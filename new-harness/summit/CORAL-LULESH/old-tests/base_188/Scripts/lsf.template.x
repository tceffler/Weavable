#! /bin/bash -l
#BSUB -q __batchqueue__
#BSUB -J __jobname__
#BSUB -o __resultsdir__/__jobname__.o%J
#BSUB -e __resultsdir__/__jobname__.e%J
#BSUB -W __walltime__
#BSUB -P STF006ACCEPT
#BSUB -nnodes __nodes__
#BSUB -alloc_flags "cpublink smt1 gpumps"

#-----------------------------------------------------
# Set up the environment for use of the harness
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load __nccstestharnessmodule__
module unload xalt
module load __compilermodulefile__
module load spectrum-mpi
module list

#-----------------------------------------------------
# Define some variables
#-----------------------------------------------------
EXECUTABLE="__pathtoexecutable__"
STARTINGDIRECTORY="__startingdirectory__"
WORKDIR="__workdir__"
RESULTSDIR="__resultsdir__"
UNIQUE_ID_STRING="__unique_id_string__"
INPUTDIR=${STARTINGDIRECTORY}/../Inputs

#-----------------------------------------------------
# Get the number of resource sets
#-----------------------------------------------------
if [ "__nodes__" -eq "188" ]
then
  nrs=3375
elif [ "__nodes__" -eq "96" ]
then
  nrs=1728
elif [ "__nodes__" -eq "12" ]
then
  nrs=216
elif [ "__nodes__" -eq "2" ]
then
  nrs=18
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
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

ulimit -s 10240
export OMP_NUM_THREADS=2

JSRUN_OPTIONS="--nrs ${nrs} --tasks_per_rs 1 --cpu_per_rs 2 --gpu_per_rs 0 -b packed:2 --stdio_mode collected --stdio_stdout stdout.txt --stdio_stderr stderr.txt"

echo jsrun ${JSRUN_OPTIONS} ${HELPER} ${EXECUTABLE} -s 64 -i 1000
jsrun ${JSRUN_OPTIONS} ${HELPER} ${EXECUTABLE} -s 64 -i 1000

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

sleep 30

#-----------------------------------------------------
# Ensure that we return to the starting directory
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
# The script now determines if we are to resubmit itself.                                            -
#-----------------------------------------------------
case __resubmitme__ in
    0) 
       test_harness_driver.py -r;;
    1) 
       echo "No resubmit";;
esac 
