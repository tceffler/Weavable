#! /bin/bash -l
#BSUB -q __batch_queue__
#BSUB -J __job_name__
#BSUB -o __results_dir__/__job_name__.o%J
#BSUB -e __results_dir__/__job_name__.e%J
#BSUB -nnodes __nodes__
#BSUB -W __walltime__
#BSUB -P __project_id__
#BSUB -alloc_flags __alloc_flags__

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#                                                    -
#-----------------------------------------------------
module load pgi/18.7
module load cuda/9.1.85
module list

ulimit -s 10240

export OMP_NUM_THREADS=__threads__
export CUDA_LAUNCH_BLOCKING=0

export OMP_STACKSIZE=64M
export PAMI_ENABLE_STRIPING=1

#-----------------------------------------------------
# Define some variables.                             -
#                                                    -
#-----------------------------------------------------
EXECUTABLE="__build_dir__/__pathtoexecutable__/__executablename__"
STARTINGDIRECTORY="__scripts_dir__"
WORKDIR="__working_dir__"
RESULTSDIR="__results_dir__"
UNIQUE_ID_STRING="__harness_id__"
INPUTFILE="__inputfile__"
EXECARGS="__execargs__"
NUM_RS=$(( __nodes__ * $RGT_MACHINE_GPUS_PER_NODE ))
CPUS_PER_RS=$(( $RGT_MACHINE_CPUS_PER_NODE / $RGT_MACHINE_GPUS_PER_NODE ))
echo "NUM_RS = $NUM_RS"
echo "CPUS_PER_RS = $CPUS_PER_RS"

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
#  Copy input files to the working directory.        -
#-----------------------------------------------------
cp $STARTINGDIRECTORY/Inputs/$INPUTFILE .

#-----------------------------------------------------
# Run the executable.                                -
#                                                    -
#-----------------------------------------------------
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

echo "jsrun --progress ./progress_UMT.${LSB_JOBID} --stdio_mode=prepend -D CUDA_VISIBLE_DEVICES -E OMP_NUM_THREADS=__threads__ --nrs __total_processes__  --tasks_per_rs 1 --cpu_per_rs 7 --gpu_per_rs 1 --bind=proportional-packed:7 -d plane:1 $EXECUTABLE $INPUTFILE $EXECARGS 1> stdout.txt 2> stderr.txt"
jsrun --progress ./progress_UMT.${LSB_JOBID} --stdio_mode=prepend -D CUDA_VISIBLE_DEVICES -E OMP_NUM_THREADS=__threads__ --nrs __total_processes__  --tasks_per_rs 1 --cpu_per_rs 7 --gpu_per_rs 1 --bind=proportional-packed:7 -d plane:1 $EXECUTABLE $INPUTFILE $EXECARGS 1> stdout.txt 2> stderr.txt

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

#-----------------------------------------------------
# Enusre that we return to the starting directory.   -
#                                                    -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR           -
#                                                    -
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR && rm -rf $WORKDIR

#-----------------------------------------------------
# Move the batch file name to  $RESULTSDIR           -
#                                                    -
#-----------------------------------------------------
mv __batchfilename__ $RESULTSDIR

#-----------------------------------------------------
# Check the final results.                           -
#                                                    -
#-----------------------------------------------------
check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

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
