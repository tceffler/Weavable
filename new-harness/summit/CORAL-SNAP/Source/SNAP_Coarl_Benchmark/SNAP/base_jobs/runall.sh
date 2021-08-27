#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J snap 
#BSUB -nnodes 192
#BSUB -alloc_flags "smt1 cpublink"
#BSUB -P stf006accept
#BSUB -q tested
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
cp ../inputs/input.96x84_516x480x504 .
cp ../inputs/h6.sh .
cp ../inputs/numactl .

# run the job
#----------------------------------------------------------------------
tstamp=`date +%m_%d_%H_%M_%S`

export OMP_NUM_THREADS=1
export BIND_SLOTS=4
export RANKS_PER_NODE=42
export BIND_THREADS=yes
export SYSTEM_CORES=2
export DISABLE_ASLR=yes
export OMPI_MCA_pml_pami_local_eager_limit=16001
export OMPI_LD_PRELOAD_POSTPEND=/ccs/home/walkup/mpitrace/spectrum_mpi/libmpihpm.so 

 jsrun -X 1 \
 --nrs 192  --tasks_per_rs 42 --cpu_per_rs 42 \
 -d plane:42 --bind rs \
 ./h6.sh  ../../CPU/snap ./input.96x84_516x480x504  out.cpu.$tstamp

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
