#! /bin/bash -l
#------------------------------------------------------------------------------
#BSUB -J gups_test_0001node
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0001node/Run_Archive/1611673226.5760067/gups_test_0001node.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0001node/Run_Archive/1611673226.5760067/gups_test_0001node.e%J
#BSUB -P csc425
#BSUB -W 10
#BSUB -q batch
#BSUB -nnodes 1
#BSUB -env "all"


# #BSUB -env "all, JOB_FEATURE=gpudefault, JOB_FEATURE=smt1"
# DEFUNCT -- #BSUB -n 6
# DEFUNCT -- #BSUB -x
# DEFUNCT -- #BSUB -R "span[ptile=6]"
# DEFUNCT -- #BSUB -b 2021:01:26:10:00

#-----------------------------------------------------
# Define variables.
#-----------------------------------------------------
STARTING_DIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0001node/Scripts"
RESULTS_DIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0001node/Run_Archive/1611673226.5760067"
WORK_DIR="/gpfs/alpine/scratch/bzf/csc425/gups/gups/test_0001node/1611673226.5760067/workdir"
WORKSPACE_DIR="/gpfs/alpine/scratch/bzf/csc425/gups/gups/test_0001node/1611673226.5760067"
EXECUTABLE_PATH="/gpfs/alpine/scratch/bzf/csc425/gups/gups/test_0001node/1611673226.5760067/build_directory/bin/gups.summit"
TEST_ID="1611673226.5760067"
BATCH_FILE_NAME="batchscript_gups_test_0001node.sh"
SOURCE_DIR="$STARTING_DIR/../../Source"
PROGRAM="gups.summit"

#------------------------------------------------------------------------------
function main
{
  #-----------------------------------------------------
  # Set up the environment for use of the harness.
  #-----------------------------------------------------
  source 
  if [ "olcf_harness_summit" != "" ] ; then
    for MYMODULE in olcf_harness_summit ; do
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
  cp -p "$WORKSPACE_DIR/build_directory/bin/$PROGRAM" $WORK_DIR
  if [ -e $WORKSPACE_DIR/$BATCH_FILE_NAME ] ; then
    cp $WORKSPACE_DIR/$BATCH_FILE_NAME $WORK_DIR
  fi
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

  jsrun  --nrs 6 --rs_per_host 6 --bind rs --cpu_per_rs 7 -g 1 --smpiargs none --stdio_mode individual  --stdio_stdout %h.%t.out --stdio_stderr %h.%t.err $EXECUTABLE_PATH  -d 1 -W 2 -s 32 -m R -g S -W 1 -R 5  -t M -u 8 -b 8589934592  1> std.out.txt 2> std.err.txt

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
                   | grep -v '^'$PROGRAM'$')"

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
  if [ "0" = 0 ] ; then
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
