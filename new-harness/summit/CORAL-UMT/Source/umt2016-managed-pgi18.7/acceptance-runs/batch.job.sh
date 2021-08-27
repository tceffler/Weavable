#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J umt
#BSUB -nnodes 192
###BSUB -alloc_flags gpumps
#BSUB -alloc_flags smt2
#BSUB -P VEN201
##BSUB -q batch
#BSUB -q tested
#BSUB -core_isolation 1
#BSUB -W 60
#---------------------------------------

if [ -z "$JOB_STREAM" ]; then
  echo "error : you must set env variable JOB_STREAM ... exiting"
  exit 0;
fi

if [ -z $MAX_JOBS ]; then
    echo "MAX_JOBS is not set - defaulting to 1"
    export MAX_JOBS=1
fi

if [ -z $JOB_NUMBER ]; then
    echo "JOB_NUMBER is not set - defaulting to 1"
    export JOB_NUMBER=1
fi
  
# optionally create a STOP file in the working directory to stop further jobs
if [ -f STOP ] ; then
    echo  "Terminating the sequence at job number $JOB_NUMBER"
    exit 0
fi

echo "starting job $JOB_NUMBER in stream $JOB_STREAM at " Mon Aug 13 15:14:30 EDT 2018

# make a directory for each stream and job; cd there to launch the job
mkdir -p stream${JOB_STREAM}_job${JOB_NUMBER}
cd stream${JOB_STREAM}_job${JOB_NUMBER}

# setup any files that are needed ...
cp ../inputs/8x12x12_38.cmg .
cp ../inputs/profile-helper.sh .

# run the job

tstamp=`date +%m_%d_%H_%M_%S`



ulimit -s 10240

export OMP_NUM_THREADS=14
export CUDA_LAUNCH_BLOCKING=0

export OMP_STACKSIZE=64M
export PAMI_ENABLE_STRIPING=1

#export LD_PRELOAD=/gpfs/wscgpfs01/walkup/mpitrace/spectrum_mpi/libhpmprof.so
#export SAVE_LIST="1"

export PROFILE_RANK=-1
export PROFILE_PATH="./nvp192_1.prof"

echo "polar = 8"
echo "azim = 4"
echo "groups = 16"

#jsrun --stdio_mode=prepend -D CUDA_VISIBLE_DEVICES -E OMP_NUM_THREADS=14 --nrs 1152  --tasks_per_rs 1 --cpu_per_rs 7   --gpu_per_rs 1 --bind=proportional-packed:7 -d plane:1 ./profile-helper.sh ../Teton/SuOlsonTest 8x12x12_38.cmg 16 2 16 8 4

status=$?

# cleanup files if needed ...

#return to the master directory and continue
cd ..

if [ $status -ne 0 ]; then
  echo "job $JOB_NUMBER returned status = $status at " $(Mon Aug 13 15:14:30 EDT 2018)
else
  echo "job $JOB_NUMBER finished at " $(`date`)
fi

# resubmit with a new job number
if [ $JOB_NUMBER -lt $MAX_JOBS ]; then
  JOB_NUMBER=$(($JOB_NUMBER+1))
  export JOB_NUMBER=$JOB_NUMBER
  sleep 5
  bsub batch.job.sh
fi


