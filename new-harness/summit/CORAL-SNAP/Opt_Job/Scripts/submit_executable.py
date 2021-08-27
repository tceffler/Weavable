#!/usr/bin/env python3

#
# Author: Arnold Tharrington
# Email: arnoldt@ornl.gov
# National Center for Computational Sciences, Scientific Computing Group.
#

import os
import getopt
import sys
import re
import time
import subprocess
import shlex

from pathlib import Path

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
    batchfilename = make_batch_script(batch_recursive_mode,path_to_workspace,test_id_string,scheduler)

    #
    # Submit the batch file to the scheduler.
    #
    sched_job_id = send_to_scheduler(path_to_workspace,test_id_string)
    print("Job id =" + str(sched_job_id))


    #
    #Write pbs job id to job_id.txt in the Status dir.
    #
    write_job_id_to_status(sched_job_id,test_id_string)


def make_batch_script(batch_recursive_mode,path_to_workspace,test_id_string,scheduler):
    batchfilename = None
    return batchfilename


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
    #
    fileobj = open(path1,"w")
    string1 = "%20s\n" % (sched_job_id)
    fileobj.write(string1)
    fileobj.close()

    return path1


def send_to_scheduler(path_to_workspace,test_id_string):
    # Form the path where the CORAL Benchmark run directory.
    #path_to_base_jobs = os.path.join(path_to_workspace,"build_directory/SNAP/opt_jobs")
    #os.chdir(path_to_base_jobs) 

    # Execute the command as a subprocess
    #command = "startall.sh"
    run_script = Path(path_to_workspace) / "build_directory/SNAP/opt_jobs/startall.sh"
    #args = shlex.split(command)

    my_stdout = None
    my_stderr = None
    #print(path_to_base_jobs)
    #print(command)

    p = subprocess.Popen(str(run_script),
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    (my_stdout, my_stderr) = p.communicate(timeout=120) 

    jobid = extract_jobid(my_stdout.decode('utf-8'))

    # Change back to the starting directory
    #os.chdir(starting_directory)

    return jobid
    

def extract_jobid(records):
    print(records)
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
