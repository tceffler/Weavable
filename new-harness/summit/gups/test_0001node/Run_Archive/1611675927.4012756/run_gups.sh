#! /bin/bash -l
#------------------------------------------------------------------------------
#BSUB -J __jobname__
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0001node/Run_Archive/1611675927.4012756/__jobname__.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0001node/Run_Archive/1611675927.4012756/__jobname__.e%J
#BSUB -P __pbsaccountid__
#BSUB -W 30
#BSUB -q __batchqueue__
#BSUB -nnodes 1
#BSUB -env "all"
__batch_file_header__

# #BSUB -env "all, JOB_FEATURE=gpudefault, JOB_FEATURE=smt1"
# DEFUNCT -- #BSUB -n __ranks__
# DEFUNCT -- #BSUB -x
# DEFUNCT -- #BSUB -R "span[ptile=__ranks_per_node__]"
# DEFUNCT -- #BSUB -b __starttime__

#-----------------------------------------------------
# Define variables.
#-----------------------------------------------------
STARTING_DIR="__starting_dir__"
RESULTS_DIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/gups/test_0001node/Run_Archive/1611675927.4012756"
WORK_DIR="__work_dir__"
WORKSPACE_DIR="__workspace_dir__"
EXECUTABLE_PATH="gups.summit"
TEST_ID="__test_id__"
BATCH_FILE_NAME="__batch_file_name__"
SOURCE_DIR="$STARTING_DIR/../../Source"
#PROGRAM="__program__"

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

  __execution_command__

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
