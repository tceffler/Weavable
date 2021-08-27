# There are 21 (of 22) cores available to the application per socket
# Each core can go up to smt4 (smt2 gives best perf for umt).
nodes=192
gpus_per_socket=3 # number of gpus to use per socket
ranks_per_gpu=1 # ranks per gpu. If greater than 1, should use mps.
let ranks_per_socket=$gpus_per_socket*$ranks_per_gpu # needs to be evenly divisible by gpus_per_socket. 
let cores_per_rank=21/$ranks_per_socket # 21 avail cores divided into the ranks.
let nmpi=2*$ranks_per_socket*$nodes  # total number of mpi ranks
let cores_per_socket=$cores_per_rank*$ranks_per_socket # this is used cores per socket (not necessarily 21)
let num_sockets=$nodes*2 #nmpi/ranks_per_socket # total number of sockets
let threads_per_rank=2*$cores_per_rank

let res_sets=2*$ranks_per_socket*$nodes

echo "nodes = $nodes"
echo "gpus used per socket = $gpus_per_socket"
echo "ranks_per_socket = $ranks_per_socket"
echo "cores_per_rank = $cores_per_rank"
echo "used cores per socket = $cores_per_socket"
echo "threads per rank = $threads_per_rank"
#--------------------------------------
grid=8x12x12_38.cmg
order=16
groups=16
type=2
polar=8
azim=4
#--------------------------------------
cat >batch.job.sh <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J umt
#BSUB -nnodes ${nodes}
###BSUB -alloc_flags gpumps
#BSUB -alloc_flags smt2
#BSUB -P VEN201
##BSUB -q batch
#BSUB -q tested
#BSUB -core_isolation 1
#BSUB -W 60
#---------------------------------------

if [ -z "\$JOB_STREAM" ]; then
  echo "error : you must set env variable JOB_STREAM ... exiting"
  exit 0;
fi

if [ -z \$MAX_JOBS ]; then
    echo "MAX_JOBS is not set - defaulting to 1"
    export MAX_JOBS=1
fi

if [ -z \$JOB_NUMBER ]; then
    echo "JOB_NUMBER is not set - defaulting to 1"
    export JOB_NUMBER=1
fi
  
# optionally create a STOP file in the working directory to stop further jobs
if [ -f STOP ] ; then
    echo  "Terminating the sequence at job number \$JOB_NUMBER"
    exit 0
fi

echo "starting job \$JOB_NUMBER in stream \$JOB_STREAM at " `date`

# make a directory for each stream and job; cd there to launch the job
mkdir -p stream\${JOB_STREAM}_job\${JOB_NUMBER}
cd stream\${JOB_STREAM}_job\${JOB_NUMBER}

# setup any files that are needed ...
cp ../inputs/$grid .
cp ../inputs/profile-helper.sh .

# run the job

tstamp=\`date +%m_%d_%H_%M_%S\`



ulimit -s 10240

export OMP_NUM_THREADS=$threads_per_rank
export CUDA_LAUNCH_BLOCKING=0

export OMP_STACKSIZE=64M
export PAMI_ENABLE_STRIPING=1

#export LD_PRELOAD=/gpfs/wscgpfs01/walkup/mpitrace/spectrum_mpi/libhpmprof.so
#export SAVE_LIST="1"

export PROFILE_RANK=-1
export PROFILE_PATH="./nvp192_1.prof"

echo "polar = $polar"
echo "azim = $azim"
echo "groups = $groups"

#jsrun --stdio_mode=prepend -D CUDA_VISIBLE_DEVICES -E OMP_NUM_THREADS=$threads_per_rank --nrs ${res_sets}  --tasks_per_rs 1 --cpu_per_rs ${cores_per_rank}   --gpu_per_rs 1 --bind=proportional-packed:${cores_per_rank} -d plane:1 ./profile-helper.sh ../Teton/SuOlsonTest $grid $groups $type $order $polar $azim

status=\$?

# cleanup files if needed ...

#return to the master directory and continue
cd ..

if [ \$status -ne 0 ]; then
  echo "job \$JOB_NUMBER returned status = \$status at " \$(`date`)
else
  echo "job \$JOB_NUMBER finished at " \$(\`date\`)
fi

# resubmit with a new job number
if [ \$JOB_NUMBER -lt \$MAX_JOBS ]; then
  JOB_NUMBER=\$((\$JOB_NUMBER+1))
  export JOB_NUMBER=\$JOB_NUMBER
  sleep 5
  bsub batch.job.sh
fi


EOF
#---------------------------------------
bsub  batch.job.sh
