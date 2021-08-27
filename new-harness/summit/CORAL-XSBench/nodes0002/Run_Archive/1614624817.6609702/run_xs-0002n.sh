#!/bin/bash -l
#BSUB -q batch
#BSUB -J XSBench
#BSUB -o /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-XSBench/nodes0002/Run_Archive/1614624817.6609702/XSBench.o%J
#BSUB -e /autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-XSBench/nodes0002/Run_Archive/1614624817.6609702/XSBench.e%J
#BSUB -nnodes 2
#BSUB -W 60
#BSUB -P csc425

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#-----------------------------------------------------
module list

#-----------------------------------------------------
# Define some variables.                             -
#-----------------------------------------------------
EXECUTABLE="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-XSBench/nodes0002/1614624817.6609702/build_directory/bin/XSBench"
STARTINGDIRECTORY="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-XSBench/nodes0002/Scripts"
WORKDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/Scratch/CORAL-XSBench/nodes0002/1614624817.6609702/workdir"
RESULTSDIR="/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-XSBench/nodes0002/Run_Archive/1614624817.6609702"
UNIQUE_ID_STRING="1614624817.6609702"

#-----------------------------------------------------
# Ensure that we are in the correct starting         -
# directory.                                         -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Make the working scratch space directory.          -
#-----------------------------------------------------
if [ ! -e $WORKDIR ]
then
    mkdir -p $WORKDIR
fi

#-----------------------------------------------------
# Make the results directory.                        -
#-----------------------------------------------------
if [ ! -e $RESULTSDIR ]
then
    mkdir -p $RESULTSDIR
fi

#-----------------------------------------------------
#  Change directory to the working directory.        -
#-----------------------------------------------------
cd $WORKDIR

#-----------------------------------------------------
# Run the executable.                                -
#-----------------------------------------------------

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

echo "jsrun -n 2 -r 1 $EXECUTABLE 1> stdout.txt 2> stderr.txt"
jsrun -n 2 -r 1 $EXECUTABLE  1> stdout.txt 2> stderr.txt

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

#-----------------------------------------------------
# Enusre that we return to the starting directory.   -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR           -
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR && rm -rf $WORKDIR

#-----------------------------------------------------
# Move the batch file name to  $RESULTSDIR           -
#-----------------------------------------------------
mv __batch_file_name__ $RESULTSDIR

#-----------------------------------------------------
# Check the final results.                           -
#-----------------------------------------------------
check_executable_driver.py -p $RESULTSDIR -i $UNIQUE_ID_STRING

#-----------------------------------------------------
# The script now determines if we are to resubmit    -
# itself.                                            -
#-----------------------------------------------------
case 1 in
    0) 
       test_harness_driver.py -r;;

    1) 
       echo "No resubmit";;
esac 

