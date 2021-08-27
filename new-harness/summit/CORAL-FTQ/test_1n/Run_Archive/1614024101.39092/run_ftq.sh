#! /bin/bash -l
#BSUB -q batch
#BSUB -J CORAL-FTQ-serial-1nodes
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-FTQ/test_1n/Run_Archive/1614024101.39092/CORAL-FTQ-serial-1nodes.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-FTQ/test_1n/Run_Archive/1614024101.39092/CORAL-FTQ-serial-1nodes.e%J
#BSUB -nnodes 1
#BSUB -W 15
#BSUB -P csc425

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#                                                    -
#-----------------------------------------------------
module load xl
module list 2>&1

#-----------------------------------------------------
# Define some variables.                             -
#                                                    -
#-----------------------------------------------------
EXECUTABLE="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-FTQ/test_1n/1614024101.39092/build_directory/ftqV110/ftq/fwq"
STARTINGDIRECTORY="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-FTQ/test_1n/Scripts"
WORKDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-FTQ/test_1n/1614024101.39092/workdir"
RESULTSDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-FTQ/test_1n/Run_Archive/1614024101.39092"
UNIQUE_ID_STRING="1614024101.39092"

#-----------------------------------------------------
# Enusre that we are in the correct starting         -
# directory.                                         -
#                                                    -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Make the working scratch space directory.          -
#                                                    -
#-----------------------------------------------------
if [ ! -e $WORKDIR ]
then
    mkdir -p $WORKDIR
fi

#-----------------------------------------------------
# Make the results directory.                        -
#                                                    -
#-----------------------------------------------------
if [ ! -e $RESULTSDIR ]
then
    mkdir -p $RESULTSDIR
fi

#-----------------------------------------------------
#  Change directory to the working directory.        -
#                                                    -
#-----------------------------------------------------
cd $WORKDIR

cp $STARTINGDIRECTORY/* .

#-----------------------------------------------------
# Run the executable.                                -
#                                                    -
#-----------------------------------------------------

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

./run.sh ${EXECUTABLE} 1 42 2>&1 | tee ./fwq-serial-run.log

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

echo "Done."

#-----------------------------------------------------
# Ensure that we return to the starting directory.   -
#                                                    -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR           -
#                                                    -
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR && rm -rf $WORKDIR

#-----------------------------------------------------
# Check the final results.                           -
#                                                    -
#-----------------------------------------------------
check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

#-----------------------------------------------------
# The script now determines if we are to resubmit    -
# itself.                                            -
#                                                    -
#-----------------------------------------------------
case 0 in
    0) 
       test_harness_driver.py -r;;

    1) 
       echo "No resubmit";;
esac 
