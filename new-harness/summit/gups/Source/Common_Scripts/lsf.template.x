#! /bin/bash -l
#------------------------------------------------------------------------------
#BSUB -J __job_name__
#BSUB -o __results_dir__/__job_name__.o%J
#BSUB -e __results_dir__/__job_name__.e%J
#BSUB -P __project_id__
#BSUB -W __walltime__
#BSUB -q __batch_queue__
#BSUB -nnodes __nodes__
#BSUB -env "all"

#-----------------------------------------------------
# Define variables.
#-----------------------------------------------------
STARTING_DIR="__scripts_dir__"
RESULTS_DIR="__results_dir__"
WORK_DIR="__working_dir__"
EXECUTABLE_PATH="__executable_path__"
TEST_ID="__harness_id__"
SOURCE_DIR="$STARTING_DIR/../../Source"
RUNTIME="__walltime__"
PPN="__processes_per_node__"
GPUS=`echo "6 / $PPN" | bc`
CPUS=`echo "1 + $GPUS" | bc`
TOTAL_PROCESS="__total_processes__"

#------------------------------------------------------------------------------
function main
{
  #-----------------------------------------------------
  # Set up the environment for use of the harness.
  #-----------------------------------------------------
  source __rgtenvironmentalfile__
  if [ "__requiredmodules__" != "" ] ; then
    for MYMODULE in __requiredmodules__ ; do
      module load $MYMODULE
    done
  fi
  module list

  #-----------------------------------------------------
  # Begin at correct starting directory (scripts dir).
  #-----------------------------------------------------
  cd $STARTING_DIR
  echo "Starting directory is $(pwd)"

  #-----------------------------------------------------
  # Copy needed files to work dir.
  #-----------------------------------------------------
  #cp $SOURCE_DIR/exec_helpers'/'* $WORK_DIR
  #cp -p "$WORKSPACE_DIR/build_directory/bin/$PROGRAM" $WORK_DIR
  #if [ -e $WORKSPACE_DIR/$BATCH_FILE_NAME ] ; then
  #  cp $WORKSPACE_DIR/$BATCH_FILE_NAME $WORK_DIR
  #fi
  env | sort > $WORK_DIR/env.txt

  #-----------------------------------------------------
  #  Change directory to the working directory.
  #-----------------------------------------------------
  cd $WORK_DIR

  #-----------------------------------------------------
  # Run the executable.
  #-----------------------------------------------------
  ulimit -s 10240

  log_binary_execution_time.py --scriptsdir $STARTING_DIR \
                         --uniqueid $TEST_ID --mode start

 CMD="jsrun --smpiargs='-disable_gpu_hooks' -g $GPUS -a 1 -n $TOTAL_PROCESS -c $CPUS $SOURCE_DIR/bin/$EXECUTABLE_PATH -e 6 -R $RUNTIME"
 #CMD="jsrun -p __total_processes__ $BUILD_DIR/$EXECUTABLE 1> run-stdout.txt 2> run-stderr.txt"
 echo "$CMD"
 $CMD


  log_binary_execution_time.py --scriptsdir $STARTING_DIR \
                         --uniqueid $TEST_ID --mode final

  # Wait for filesystem to quiesce
  sleep 30
  #sync

  #-----------------------------------------------------
  # Copy the results back to the $RESULTS_DIR.
  #-----------------------------------------------------
  ARCHIVE_FILES="$(find . -name '*' -type f -print \
                   | grep -v 'core\.[0-9][0-9]*$' \
                   | grep -v '^'$EXECUTABLE_PATH'$')"

  tar cf - $ARCHIVE_FILES | ( cd $RESULTS_DIR ; tar xf - )

  #-----------------------------------------------------
  # Return to the starting directory.
  #-----------------------------------------------------
  cd $STARTING_DIR

  #-----------------------------------------------------
  # Check the final results.
  #-----------------------------------------------------
  check_executable_driver.py -p $RESULTS_DIR -i $TEST_ID

  #-----------------------------------------------------
  # Determine whether to resubmit.
  #-----------------------------------------------------
  if [ "__resubmitme__" = 0 ] ; then
    test_harness_driver.py -r
  else
    echo "No resubmit"
  fi
}
#------------------------------------------------------------------------------

[ -d $WORK_DIR ] || mkdir -p $WORK_DIR
#[ -d $RESULTS_DIR ] || mkdir -p $RESULTS_DIR

main $@ 1> $RESULTS_DIR/job_out.txt 2> $RESULTS_DIR/job_err.txt

#------------------------------------------------------------------------------
