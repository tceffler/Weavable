#! /bin/bash -l
#BSUB -q __batchqueue__
#BSUB -J __jobname__-__total_nodes__nodes
#BSUB -o __resultsdir__/__jobname__-__total_nodes__nodes.o%J
#BSUB -e __resultsdir__/__jobname__-__total_nodes__nodes.e%J
#BSUB -nnodes __total_nodes__
#BSUB -W 15
#BSUB -P __projectid__

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#                                                    -
#-----------------------------------------------------
source __rgtenvironmentalfile__
module load olcf_harness_summit

module load xl
module list 2>&1

#-----------------------------------------------------
# Define some variables.                             -
#                                                    -
#-----------------------------------------------------
EXECUTABLE="ftq"
STARTINGDIRECTORY="__startingdirectory__"
WORKDIR="__workdir__"
RESULTSDIR="__resultsdir__"
UNIQUE_ID_STRING="__unique_id_string__"
BUILDDIR=`dirname $WORKDIR`/build_directory

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

./run.sh ${EXECUTABLE} __total_nodes__ 42 2>&1 | tee ./fwq-serial-run.log

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
