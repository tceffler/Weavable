#!/bin/bash
#--------------------------------------------------------------------------------
# mpirun -np $nmpi helper.sh your.exe [args]
# optionally set BIND_SLOTS in your job script = #hwthreads per rank
# Note : for some OpenMP implementations (GNU OpenMP) use mpirun --bind-to none
#--------------------------------------------------------------------------------
let ngpus=4

    cpus_per_node=152
    declare -a list0=(`seq 0 75`)
    declare -a list1=(`seq 88 163`)
    declare -a mpscpu=(76-79 80-83 164-167 168-171)
    declare -a gpulist=(0 1 3 4)

#--------------------------------------------
# handle jobs that don't use MPI
#--------------------------------------------
if [ -z "$OMPI_COMM_WORLD_LOCAL_SIZE" ]; then
  let OMPI_COMM_WORLD_LOCAL_SIZE=1
  let OMPI_COMM_WORLD_LOCAL_RANK=0
fi

#---------------------------------------------
# set CUDA device for each MPI rank
#---------------------------------------------
let product=$ngpus*$OMPI_COMM_WORLD_LOCAL_RANK
let gpu=$product/$OMPI_COMM_WORLD_LOCAL_SIZE
let mydevice=${gpulist[$gpu]}

echo rank $OMPI_COMM_WORLD_RANK $OMPI_COMM_WORLD_LOCAL_RANK " using device " $mydevice 

#---------------------------------------------
# optionally start MPS
#---------------------------------------------
if [ "$USE_MPS" == "yes" ]; then
  if [ $OMPI_COMM_WORLD_LOCAL_RANK = 0 ]; then
    if [ $OMPI_COMM_WORLD_RANK = 0 ]; then
      echo starting mps ...
    fi  
    for ((g=0; g<$ngpus; g++))
    do  
     let i=${gpulist[$g]}
     rm -rf /dev/shm/${USER}/mps_$i
     rm -rf /dev/shm/${USER}/mps_log_$i
     mkdir -p /dev/shm/${USER}/mps_$i
     mkdir -p /dev/shm/${USER}/mps_log_$i
     echo CUDA_VISIBLE_DEVICES=$i
     export CUDA_VISIBLE_DEVICES=$i
     export CUDA_MPS_PIPE_DIRECTORY=/dev/shm/${USER}/mps_$i
     export CUDA_MPS_LOG_DIRECTORY=/dev/shm/${USER}/mps_log_$i
     echo taskset -c ${mpscpu[$g]} /usr/bin/nvidia-cuda-mps-control -d
     taskset -c ${mpscpu[$g]} /usr/bin/nvidia-cuda-mps-control -d
     sleep 1
    done
#     rm -rf /dev/shm/${USER}/mps
#     rm -rf /dev/shm/${USER}/mps_log
#     mkdir -p /dev/shm/${USER}/mps
#     mkdir -p /dev/shm/${USER}/mps_log
#     export CUDA_MPS_PIPE_DIRECTORY=/dev/shm/${USER}/mps
#     export CUDA_MPS_LOG_DIRECTORY=/dev/shm/${USER}/mps_log
#     echo taskset -c ${mpscpu[0]} /usr/bin/nvidia-cuda-mps-control -d
#     taskset -c ${mpscpu[0]} /usr/bin/nvidia-cuda-mps-control -d
#     sleep 1
  else
    sleep $ngpus
  fi
  sleep 1
  printf -v myfile "/dev/shm/${USER}/mps_%d" $mydevice
  export CUDA_MPS_PIPE_DIRECTORY=$myfile
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES=$mydevice
fi

#-------------------------------------------------
# assign socket and affinity mask
#-------------------------------------------------
let x2rank=2*$OMPI_COMM_WORLD_LOCAL_RANK
let socket=$x2rank/$OMPI_COMM_WORLD_LOCAL_SIZE
let ranks_per_socket=$OMPI_COMM_WORLD_LOCAL_SIZE/2

# divide available slots evenly or specify slots by env variable
if [ -z "$BIND_SLOTS" ]; then
  let cpus_per_rank=$cpus_per_node/$OMPI_COMM_WORLD_LOCAL_SIZE
else
  let cpus_per_rank=$BIND_SLOTS 
fi

if [ -z "$OMP_NUM_THREADS" ]; then
  let OMP_NUM_THREADS=$cpus_per_rank
fi

# BIND_STRIDE is used in OMP_PLACES ... it will be 1 if OMP_NUM_THREADS was not set
let BIND_STRIDE=$cpus_per_rank/$OMP_NUM_THREADS

if [ $socket = 0 ]; then
  let ndx=$OMPI_COMM_WORLD_LOCAL_RANK*$cpus_per_rank
  let start_cpu=${list0[$ndx]}
  let stop_cpu=$start_cpu+$cpus_per_rank-1
else
  let rank_in_socket=$OMPI_COMM_WORLD_LOCAL_RANK-$ranks_per_socket
  let ndx=$rank_in_socket*$cpus_per_rank
  let start_cpu=${list1[$ndx]}
  let stop_cpu=$start_cpu+$cpus_per_rank-1
fi

#---------------------------------------------
# set OMP_PLACES or GOMP_CPU_AFFINITY
#---------------------------------------------
if [ "$USE_GOMP" == "yes" ]; then
  export GOMP_CPU_AFFINITY="$start_cpu-$stop_cpu:$BIND_STRIDE"
else
  export OMP_PLACES={$start_cpu:$OMP_NUM_THREADS:$BIND_STRIDE}
fi

#-------------------------------------------------
# set an affinity mask for each rank using taskset
#-------------------------------------------------
printf -v command "taskset -c %d-%d"  $start_cpu  $stop_cpu 

executable=$1

shift

#-------------------------
# run the code
#-------------------------
$command $executable "$@"

#---------------------------------------------
# optionally stop MPS 
#---------------------------------------------
if [ "$USE_MPS" == "yes" ]; then
  if [ $OMPI_COMM_WORLD_LOCAL_RANK = 0 ]; then
    if [ $OMPI_COMM_WORLD_RANK = 0 ]; then
      echo stopping mps ...
    fi
    sleep 1
    for ((g=0; g<$ngpus; g++))
    do
     let i=${gpulist[$g]}
     export CUDA_MPS_PIPE_DIRECTORY=/dev/shm/${USER}/mps_$i
     echo "quit" | /usr/bin/nvidia-cuda-mps-control
     sleep 1
     rm -rf /dev/shm/${USER}/mps_$i
     rm -rf /dev/shm/${USER}/mps_log_$i
    done
#     echo "quit" | /usr/bin/nvidia-cuda-mps-control
#     sleep 1
#     rm -rf /dev/shm/${USER}/mps
#     rm -rf /dev/shm/${USER}/mps_log
#    rm -rf /dev/shm/${USER}
    unset CUDA_MPS_PIPE_DIRECTORY
  fi
fi
