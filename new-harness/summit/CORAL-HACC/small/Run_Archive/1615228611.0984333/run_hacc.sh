#! /bin/bash -l
#BSUB -q batch
#BSUB -J hacc_small
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-HACC/small/Run_Archive/1615228611.0984333/hacc_small.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-HACC/small/Run_Archive/1615228611.0984333/hacc_small.e%J
#BSUB -W 30
#BSUB -P CSC425
#BSUB -nnodes 2
#BSUB -alloc_flags "smt1 cpublink"

#-----------------------------------------------------
# Set up the environment for use of the harness
#-----------------------------------------------------
module load xl
module load essl
module load cuda
module load spectrum-mpi
module list

#-----------------------------------------------------
# Define some variables
#-----------------------------------------------------
EXECUTABLE="bin/hacc_tpm"
STARTINGDIRECTORY="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-HACC/small/Scripts"
WORKDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-HACC/small/1615228611.0984333/workdir"
RESULTSDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-HACC/small/Run_Archive/1615228611.0984333"
#UNIQUE_ID_STRING="__unique_id_string__"
UNIQUE_ID_STRING="1615228611.0984333"
INPUTDIR=${STARTINGDIRECTORY}/../Inputs

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

echo jsrun --smpiargs="-mca coll ^ibm" ${JSRUN_OPTIONS} $EXECUTABLE ./indat ./cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 512 -t 2x2x2
time jsrun --smpiargs="-mca coll ^ibm" ${JSRUN_OPTIONS} $EXECUTABLE ./indat ./cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 512 -t 2x2x2

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
mv run_HACC.sh $RESULTSDIR

#-----------------------------------------------------
# Check the final results
#-----------------------------------------------------
check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

#-----------------------------------------------------
# The script now determines if we are to resubmit itself                                           -
#-----------------------------------------------------
case 0 in
    0) 
       test_harness_driver.py -r;;
    1) 
       echo "No resubmit";;
esac 
