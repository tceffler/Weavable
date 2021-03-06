#! /bin/bash -l
#BSUB -J __jobname__
#BSUB -o __resultsdir__/__jobname__.o%J
#BSUB -e __resultsdir__/__jobname__.e%J
#BSUB -nnodes __nodes__
#BSUB -alloc_flags __alloc_flags__
#BSUB -P __projectid__
#BSUB -q __batchqueue__
#BSUB -W __walltime__

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load __nccstestharnessmodule__
module load cuda
module list

export OMP_NUM_THREADS=__threads__

#-----------------------------------------------------
# Define some variables.                             -
#-----------------------------------------------------
EXECUTABLE="__executable_path__"
STARTINGDIRECTORY="__scripts_dir__"
WORKDIR="__working_dir__"
RESULTSDIR="__results_dir__"
UNIQUE_ID_STRING="__harness_id__"
INPUTFILE="__inputfile__"
EXECARGS="__execargs__"
NUM_RS=$(( __nodes__ * $RGT_MACHINE_GPUS_PER_NODE ))
CPUS_PER_RS=$(( $RGT_MACHINE_CPUS_PER_NODE / $RGT_MACHINE_GPUS_PER_NODE ))
PPRS=4
echo "NUM_RS = $NUM_RS"
echo "CPUS_PER_RS = $CPUS_PER_RS"
echo "PPRS = $PPRS"

#-----------------------------------------------------
# Enusre that we are in the correct starting         -
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
cp $STARTINGDIRECTORY/Inputs/$INPUTFILE .

#-----------------------------------------------------
# Run the executable.                                -
#-----------------------------------------------------
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

echo "jsrun --progress ./progress_amg.${LSB_JOBID} --nrs $NUM_RS --tasks_per_rs $PPRS --cpu_per_rs $CPUS_PER_RS -g 1 --latency_priority=gpu-cpu -D CUDA_VISIBLE_DEVICES -d plane:$PPRS -b packed:1 $EXECUTABLE $EXECARGS 1> stdout.txt 2> stderr.txt"
jsrun --progress ./progress_amg.${LSB_JOBID} --nrs $NUM_RS --tasks_per_rs $PPRS --cpu_per_rs $CPUS_PER_RS -g 1 --latency_priority=gpu-cpu -D CUDA_VISIBLE_DEVICES -d plane:$PPRS -b packed:1 $EXECUTABLE $EXECARGS 1> stdout.txt 2> stderr.txt

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
mv __batchfilename__ $RESULTSDIR

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
