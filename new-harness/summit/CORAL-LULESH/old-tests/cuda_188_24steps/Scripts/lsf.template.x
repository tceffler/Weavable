#! /bin/bash -l
#BSUB -q __batchqueue__
#BSUB -J __jobname__
#BSUB -o __resultsdir__/__jobname__.o%J
#BSUB -e __resultsdir__/__jobname__.e%J
#BSUB -W __walltime__
#BSUB -P STF006ACCEPT
#BSUB -nnodes __total_nodes__
#BSUB -alloc_flags "gpumps"

#-----------------------------------------------------
# Set up the environment for use of the harness
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load __nccstestharnessmodule__
module unload xalt
module load __compilermodulefile__
module load cuda
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

n_steps=__num_jobsteps__

echo "Running ${n_steps} jobsteps with:"
echo jsrun ${JSRUN_OPTIONS} ${HELPER} ${EXECUTABLE} -s 128 -i 5000

for i_step in `seq -w $n_steps`
do 
  launchdir=jobstep_${i_step}
  mkdir -p $launchdir && cd $launchdir
  find .. -maxdepth 1 -type f -exec cp -al {} . \;

  echo `date` ": Launching job step ${i_step} in $PWD"

  jsrun ${JSRUN_OPTIONS} ${HELPER} ${EXECUTABLE} -s 128 -i 5000 2>&1 > stderrout.${i_step}.txt &
  cd ..
done

echo `date` ": Waiting for steps ..."
#jswait all
wait
echo `date` ": All jobsteps finished."

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
