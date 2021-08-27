#! /bin/bash -l
#BSUB -q batch
#BSUB -J umt_opt_n0016
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/opt_n0192/Run_Archive/1614627734.40128/umt_opt_n0016.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/opt_n0192/Run_Archive/1614627734.40128/umt_opt_n0016.e%J
#BSUB -nnodes 192
#BSUB -W 60
#BSUB -P csc425
#BSUB -alloc_flags smt2

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#                                                    -
#-----------------------------------------------------
module load pgi/18.7
module load cuda/9.1.85
module list

ulimit -s 10240

export OMP_NUM_THREADS=14
export CUDA_LAUNCH_BLOCKING=0

export OMP_STACKSIZE=64M
export PAMI_ENABLE_STRIPING=1

#-----------------------------------------------------
# Define some variables.                             -
#                                                    -
#-----------------------------------------------------
EXECUTABLE="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-UMT/opt_n0192/1614627734.40128/build_directory/bin/SuOlsonTest"
STARTINGDIRECTORY="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/opt_n0192/Scripts"
WORKDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-UMT/opt_n0192/1614627734.40128/workdir"
RESULTSDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/opt_n0192/Run_Archive/1614627734.40128"
UNIQUE_ID_STRING="1614627734.40128"
INPUTFILE="8x12x12_38.cmg"
EXECARGS="16 2 16 8 4"
NUM_RS=$(( 192 * $RGT_MACHINE_GPUS_PER_NODE ))
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

echo "jsrun --progress ./progress_UMT.${LSB_JOBID} --stdio_mode=prepend -D CUDA_VISIBLE_DEVICES -E OMP_NUM_THREADS=14 --nrs 1152  --tasks_per_rs 1 --cpu_per_rs 7 --gpu_per_rs 1 --bind=proportional-packed:7 -d plane:1 $EXECUTABLE $INPUTFILE $EXECARGS 1> stdout.txt 2> stderr.txt"
jsrun --progress ./progress_UMT.${LSB_JOBID} --stdio_mode=prepend -D CUDA_VISIBLE_DEVICES -E OMP_NUM_THREADS=14 --nrs 1152  --tasks_per_rs 1 --cpu_per_rs 7 --gpu_per_rs 1 --bind=proportional-packed:7 -d plane:1 $EXECUTABLE $INPUTFILE $EXECARGS 1> stdout.txt 2> stderr.txt

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
case 1 in
    0) 
       test_harness_driver.py -r;;

    1) 
       echo "No resubmit";;
esac 
