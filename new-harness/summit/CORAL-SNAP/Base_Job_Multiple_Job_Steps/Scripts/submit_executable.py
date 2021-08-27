#!/usr/bin/env python3

#
# Author: Arnold Tharrington
# Email: arnoldt@ornl.gov
# National Center for Computational Sciences, Scientific Computing Group.
#

# System imports
import os
import getopt
import sys
import re
import time
import subprocess
import shlex

# Modify the python path for local LAMMPS python harness packages.
sys.path.append('../../Source/Snap_Python_Packages')

# Local imports
import job_submitter
import build_utilities
from file_parsers.snap_configuration_file import snap_configuration_file

#-----------------------------------------------------
# Define the starting directory.                     -
#                                                    -
#-----------------------------------------------------
starting_directory = os.getcwd()

def main():
    #
    # Get the command line arguments.
    #
    try:
        opts,args = getopt.getopt(sys.argv[1:],"hi:p:r")

    except getopt.GetoptError:
            usage()
            sys.exit(2)

    #
    # Parse the command line arguments.
    #
    if opts == []:
        usage()
        sys.exit()

    #
    # Initialize some variables.
    #
    batch_recursive_mode = "1"

    for o, a in opts:
        if o == "-p":
            path_to_workspace = a
        elif o == "-i":
            test_id_string = a
        elif o == "-r":
            batch_recursive_mode = "0"
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            usage()
            sys.exit()


    submit_executable(path_to_workspace,
                     test_id_string,
                     batch_recursive_mode)
    return

def submit_executable(path_to_workspace,
                      test_id_string,
                      batch_recursive_mode):
    #
    # Make the batch script.
    #
    scheduler = "lsf"
    my_job_submitter = create_a_job_submitter(batch_recursive_mode,path_to_workspace,test_id_string,scheduler)

    #
    # Submit the job to the scheduler.
    #
    submit_scheduler_return_code = my_job_submitter.submitJob(mode="Normal")
    sched_job_id = my_job_submitter.JOBID

    #
    # Write job id to job_id.txt in the Status dir.
    #
    write_job_id_to_status(sched_job_id,
                           test_id_string)

    return submit_scheduler_return_code 

def create_a_job_submitter(batch_recursive_mode,
                           path_to_workspace,
                           test_id_string,
                           scheduler):
    my_snap_configuration = snap_configuration_file("snap_configuration.txt")

    #-----------------------------------------------------
    # Define the variable resubmitme.                    -
    #                                                    -
    #-----------------------------------------------------
    if batch_recursive_mode == True:
        resubmitme = "0"
    else:
        resubmitme = "1"

    #-----------------------------------------------------
    # Define the batch file name.                        -
    #                                                    -
    # This will be the file that is submiited to the     -
    # batch system.                                      -
    #-----------------------------------------------------
    my_batchfilename = my_snap_configuration.name_of_batchfile

    #-----------------------------------------------------
    # Define the wall time of the job.                   -
    #                                                    -
    #-----------------------------------------------------
    my_wall_time = my_snap_configuration.walltime 

    #-----------------------------------------------------
    # Define the queue to run the job in.                -
    #                                                    -
    #-----------------------------------------------------
    my_batch_queue = my_snap_configuration.queue

    #-----------------------------------------------------
    # Define the project id.                             -
    #                                                    -
    #-----------------------------------------------------
    my_project_id = os.getenv('RGT_PBS_JOB_ACCNT_ID')
    nccstestharnessmodule = os.environ["RGT_NCCS_TEST_HARNESS_MODULE"]
    rgtenvironmentalfile = os.environ["RGT_ENVIRONMENTAL_FILE"]

    #-----------------------------------------------------
    # Define the number on nodes required for this       -
    # job.                                               -
    #                                                    -
    # The number_nodes_per_copy is a constant and set at -
    # 192.                                               -
    #                                                    -
    # The number_copies is a variable. The goal is to    -
    # bewteen 90% and 100% of Summit.                    -
    #                                                    -
    #-----------------------------------------------------
    number_copies = int(my_snap_configuration.number_of_copies)
    number_nodes_per_copy=192 # This is a constant. Do not modify.
    my_number_of_nodes = str(number_copies*number_nodes_per_copy)

    #-----------------------------------------------------
    # Define the job name.                               -
    #                                                    -
    #-----------------------------------------------------
    my_jobname = my_snap_configuration.name_of_jobname + "." + str(number_copies) + "x" + str(number_nodes_per_copy)

    #-----------------------------------------------------
    # Get the path to the results directory.             -
    #                                                    -
    # The results directory is where the results         -
    # of the job are stored.                             -
    #-----------------------------------------------------
    resultsdir = get_path_to_results_dir(test_id_string)


    #-----------------------------------------------------
    # Define the path to the workdir.                    -
    #                                                    -
    #-----------------------------------------------------
    workdir = os.path.join(path_to_workspace,"workdir")

    #-----------------------------------------------------
    # Get the path to the snap cpu executable.           -
    #                                                    -
    #-----------------------------------------------------
    my_executable = build_utilities.get_path_to_snap_cpu_executable(path_to_workspace)

    #-----------------------------------------------------
    # Get the dirname of the snap input files.           -
    #                                                    -
    #-----------------------------------------------------
    dirnameinputfiles = build_utilities.get_dirname_of_basejob_snap_inputfiles(path_to_workspace)

    #-----------------------------------------------------
    # Create a job submitter.                            -
    #                                                    -
    #-----------------------------------------------------
    my_job_submitter = job_submitter.SummitJobSubmitter()


    list_of_features = [ 
                ("batchqueue",re.compile("__batchqueue__"),my_batch_queue),
                ("nccstestharnessmodule",re.compile("__nccstestharnessmodule__"),nccstestharnessmodule),
                ("rgtenvironmentalfile",re.compile("__rgtenvironmentalfile__"),rgtenvironmentalfile),
                ("walltime",re.compile("__walltime__"),my_wall_time),
                ("projectid",re.compile("__projectid__"),my_project_id),
                ("nm_nodes",re.compile("__nnodes__"),my_number_of_nodes),
                ("jobname",re.compile("__jobname__"),my_jobname),
                ("resultsdir",re.compile("__resultsdir__"),resultsdir),
                ("workdir",re.compile("__workdir__"),workdir),
                ("batchfilename",re.compile("__batchfilename__"),my_batchfilename),
                ("test_id_string",re.compile("__uniqueidstring__"),test_id_string),
                ("starting_directory",re.compile("__startingdirectory__"),starting_directory),
                ("executable",re.compile("__pathtoexecutable__"),my_executable),
                ("maxcopies",re.compile("__maxcopies__"),str(number_copies)),
                ("resubmitme",re.compile("__resubmitme__"),resubmitme),
                ("dirnameinputfiles",re.compile("__dirnameinputfiles__"),dirnameinputfiles),
               ]

    my_job_submitter.initialize(batchfilename=my_batchfilename)

    for feature in list_of_features:
        my_job_submitter.addFeature(name=feature[0],
                                    value=feature[2],
                                    rgexp=feature[1])


    return my_job_submitter


def get_path_to_results_dir(test_id_string):
    #
    # Get the current working directory.
    #
    cwd = os.getcwd()

    #
    # Get the 1 head path in the cwd.
    #
    (dir_head1, dir_tail1) = os.path.split(cwd)

    #
    # Now join dir_head1 to make the path. This path should be unique.
    #
    path1 = os.path.join(dir_head1,"Run_Archive",test_id_string)

    return path1

def write_job_id_to_status(sched_job_id,test_id_string):
    #
    # Get the current working directory.
    #
    cwd = os.getcwd()

    #
    # Get the 1 head path in the cwd.
    #
    (dir_head1, dir_tail1) = os.path.split(cwd)

    #
    # Now join dir_head1 to make the path. This path should be unique.
    #
    path1 = os.path.join(dir_head1,"Status",test_id_string,"job_id.txt")

    #
    # Write the pbs job id to the file.
    
    fileobj = open(path1,"w")
    string1 = "%20s\n" % (sched_job_id)
    fileobj.write(string1)
    fileobj.close()

    return path1


def extract_jobid(records):
    words = records.split()
    job_id = words[1].replace("<","")
    job_id = job_id.replace(">","")
    return job_id

def usage():
    print("Usage: submit_executable.x [-h|--help] -p <path_to_worspace> -i <test_id_string>")
    print("")
    print("A driver program that the submits the binary thru batch for the testing.")
    print("The submit program also writes the job id of the submitted batch job to the file")
    print("'Status/<test_id_string>/job_id.txt'. The only line in job_id.txt is the job id.")
    print()
    print("-h, --help           Prints usage information.")                              
    print("-p                   The absolute path to the workspace. This path   ")
    print("                     must have the appropiate permissions to permit  ")
    print("                     the user of the test to r,w, and x.             ")
    print("-i                   The test id string. The build program           ")
    print("                     uses this string to make a unique directory     ")
    print("                     within path_to_workspace. We don't want         ")
    print("                     concurrent builds to clobber each other.        ")
    print("                     The submit program uses this string to write the")
    print("                     job schedule id to 'Status/<test_id_string>/job_id.txt.")
    print("-r                   The batch script will resubmit itself, otherwise")
    print("                     only 1 instance will be submitted               ")


if __name__ == "__main__" :
    main()
