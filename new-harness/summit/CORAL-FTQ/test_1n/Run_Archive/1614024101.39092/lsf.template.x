#! /bin/bash -l
#BSUB -q __batch_queue__
#BSUB -J __job_name__-__nodes__nodes
#BSUB -o __results_dir__/__job_name__-__nodes__nodes.o%J
#BSUB -e __results_dir__/__job_name__-__nodes__nodes.e%J
#BSUB -nnodes __nodes__
#BSUB -W __walltime__
#BSUB -P __project_id__

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
EXECUTABLE="__build_dir__/__pathtoexecutable__/__executablename__"
STARTINGDIRECTORY="__scripts_dir__"
WORKDIR="__working_dir__"
RESULTSDIR="__results_dir__"
UNIQUE_ID_STRING="__harness_id__"

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

./run.sh ${EXECUTABLE} __nodes__ __processes_per_node__ 2>&1 | tee __run_logfile__

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
case __resubmitme__ in
    0) 
       test_harness_driver.py -r;;

    1) 
       echo "No resubmit";;
esac 
