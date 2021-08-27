#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J snap 
##BSUB -nnodes 192
#BSUB -nnodes 2
#BSUB -alloc_flags "smt1 cpublink"
#BSUB -P csc425
#BSUB -q batch
#BSUB -W 35
#--------------------------------------------------------------------------------------

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

echo "starting job $JOB_NUMBER in stream $JOB_STREAM at " `date`

# make a directory for each stream and job; cd there to launch the job
mkdir -p stream${JOB_STREAM}_job${JOB_NUMBER}
cd stream${JOB_STREAM}_job${JOB_NUMBER}
# setup any files that are needed ...
cp ../inputs/input.24x32_512x384x512 .
cp ../inputs/helper.sh .
cp ../inputs/numactl .

# run the job
#----------------------------------------------------------------------
tstamp=`date +%m_%d_%H_%M_%S`

export RANKS_PER_NODE=4
export OMP_NUM_THREADS=2
export USE_GOMP=yes
export BIND_SLOTS=40
export OMPI_LD_PRELOAD_POSTPEND=/ccs/home/walkup/mpitrace/spectrum_mpi/libmpitrace.so

jsrun -X 1    \
 --nrs 192  --tasks_per_rs 4  --cpu_per_rs 42  \
 --gpu_per_rs 6  --bind proportional-packed:10 -d plane:4   \
 ./helper.sh ../../GPU/snap  ./input.24x32_512x384x512  out.gpu.$tstamp

status=$?

# cleanup files if needed ...

#return to the master directory and continue
cd ..

if [ $status -ne 0 ]; then
  echo "job $JOB_NUMBER returned status = $status at " `date`
else
  echo "job $JOB_NUMBER finished at " `date`
fi

# resubmit with a new job number
if [ $JOB_NUMBER -lt $MAX_JOBS ]; then
  JOB_NUMBER=$(($JOB_NUMBER+1))
  export JOB_NUMBER=$JOB_NUMBER
  sleep 5
  bsub -o %J.out -e %J.err -J snap -nnodes 192 -P stf006accept -alloc_flags "smt1 cpublink" -q tested -W 35  runall.sh
fi
