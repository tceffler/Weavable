#! /bin/bash -l
#BSUB -P __project_id__
#BSUB -q __batch_queue__
#BSUB -J __job_name__
#BSUB -o __results_dir__/__job_name__.o%J
#BSUB -e __results_dir__/__job_name__.e%J
#BSUB -n __number_processors__
#BSUB -W __walltime__

#-----------------------------------------------------
# Set up the environment for use of the harness.     -
#                                                    -
#-----------------------------------------------------
#source __rgtenvironmentalfile__
#module load __nccstestharnessmodule__

#-----------------------------------------------------
# Define some variables.                             -
#                                                    -
#-----------------------------------------------------
EXECUTABLE="__path_to_executable__"
STARTINGDIRECTORY="__starting_directory__"
WORKDIR="__work_dir__"
RESULTSDIR="__result_sdir__"
UNIQUE_ID_STRING="__unique_id_string__"

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

#-----------------------------------------------------
# Run the executable.                                -
#                                                    -
#-----------------------------------------------------
log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode start

__job_launch_command__

log_binary_execution_time.py --scriptsdir $STARTINGDIRECTORY --uniqueid $UNIQUE_ID_STRING --mode final

sleep 30
#-----------------------------------------------------
# Enusre that we return to the starting directory.   -
#                                                    -
#-----------------------------------------------------
cd $STARTINGDIRECTORY

#-----------------------------------------------------
# Copy the results back to the $RESULTSDIR           -
#                                                    -
#-----------------------------------------------------
cp -rf $WORKDIR/* $RESULTSDIR && rm -rf $WORKDIR

#-----------------------------------------------------
# Move the batch file name to  $RESULTSDIR           -
#                                                    -
#-----------------------------------------------------
mv __batchfilename__ $RESULTSDIR

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
