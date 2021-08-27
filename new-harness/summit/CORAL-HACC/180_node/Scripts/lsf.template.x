#! /bin/bash -l
#BSUB -q __batchqueue__
#BSUB -J __jobname__
#BSUB -o __resultsdir__/__jobname__.o%J
#BSUB -e __resultsdir__/__jobname__.e%J
#BSUB -W __walltime__
#BSUB -P STF006ACCEPT
#BSUB -nnodes __nodes__
#BSUB -alloc_flags "smt1 cpublink"

#-----------------------------------------------------
# Set up the environment for use of the harness
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load __nccstestharnessmodule__
module load xl
module load essl
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
nodes=$(( __nodes__ ))
ppn=36
nmpi=$(( __nodes__ * $ppn ))
cores_per_rank=$(( 42 / $ppn ))
cores=$(( 42 * $nodes ))
ulimit -c 0
export OMP_NUM_THREADS=1
export SAVE_LIST=0
export RANKS_PER_NODE=${ppn}
export BIND_SLOTS=4
export BIND_THREADS=yes
export SYSTEM_CORES=2
export USE_MPS=yes
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ENABLE_OOO_AR=1
export PAMI_IBV_QP_SERVICE_LEVEL=8
#export PAMI_IBV_ENABLE_DCT=1
export PAMI_IBV_CQEDEPTH=40000
export PAMI_IBV_SQDEPTH=128
export PAMI_PMIX_DATACACHE=1
export DISABLE_ASLR=yes
export OMP_TARGET_OFFLOAD=MANDATORY


#-----------------------------------------------------
# Run the executable
#-----------------------------------------------------
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start
tstamp=`date +%m_%d_%H_%M_%S`

JSRUN_OPTIONS="-X 1 --progress ${tstamp}.progress --nrs ${nodes} --tasks_per_rs ${ppn} --cpu_per_rs 42 --gpu_per_rs 6 --bind=proportional-packed:${cores_per_rank} -d plane:${ppn} --stdio_mode collected --stdio_stdout stdout.txt --stdio_stderr stderr.txt"

echo jsrun --smpiargs="-mca coll ^ibm" ${JSRUN_OPTIONS} ./h6smt1.sh $EXECUTABLE ./indat ./cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 512 -t 18x18x20
time jsrun --smpiargs="-mca coll ^ibm" ${JSRUN_OPTIONS} ./h6smt1.sh $EXECUTABLE ./indat ./cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 512 -t 18x18x20

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
