#! /bin/bash -l
#------------------------------------------------------------------------------
#BSUB -J gups
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0002node/Run_Archive/1612208917.6524537/gups.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0002node/Run_Archive/1612208917.6524537/gups.e%J
#BSUB -P csc425
#BSUB -W 60
#BSUB -q batch
#BSUB -nnodes 2
#BSUB -env "all"

#-----------------------------------------------------
# Define variables.
#-----------------------------------------------------
STARTING_DIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0002node/Scripts"
RESULTS_DIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0002node/Run_Archive/1612208917.6524537"
WORK_DIR="/gpfs/alpine/scratch/bzf/csc425/gups/gups/test_0002node/1612208917.6524537/workdir"
EXECUTABLE_PATH="gups.summit"
TEST_ID="1612208917.6524537"
SOURCE_DIR="$STARTING_DIR/../../Source"
RUNTIME="60"
PPN="1"

#------------------------------------------------------------------------------
function main
{
  #-----------------------------------------------------
  # Set up the environment for use of the harness.
  #-----------------------------------------------------
  source 
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

 CMD="jsrun --smpiargs='-disable_gpu_hooks' -g 6 -n 2 -r $PPN -c ALL_CPUS $SOURCE_DIR/bin/$EXECUTABLE_PATH -e 6 -R $RUNTIME"
 #CMD="jsrun -p 2 $BUILD_DIR/$EXECUTABLE 1> run-stdout.txt 2> run-stderr.txt"
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
