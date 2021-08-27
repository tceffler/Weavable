#! /usr/bin/env python3

## @package build_utilities
# Contains global utility definitions, classes, etc. for building and running SNAP benchmark.
#

# System imports
import os

#-----------------------------------------------------
# A hash that defines success and failure            -
# for return codes for building the various          -
# packages.                                          -
#                                                    -
#-----------------------------------------------------
unix_exit_status = {"success" : 0,
                    "failure" : 1}

#-----------------------------------------------------
# Define good and bad results.                       -
#                                                    -
#-----------------------------------------------------
verification_results = {"GOOD_RESULTS" : 1,
                        "BAD_RESULTS"  : 0}

#-----------------------------------------------------
# Define a format string for unix commands.          -
#                                                    -
#-----------------------------------------------------
subprocess_command_message  = "Execution directory = {execdir}\n"
subprocess_command_message += "Command = {command}\n"
subprocess_command_message += "Return code = {returncode}\n\n"
subprocess_command_message += "=====================\n"
subprocess_command_message += "= Below is stdout\n"
subprocess_command_message += "=====================\n"
subprocess_command_message += "\n{stdout}\n\n"
subprocess_command_message += "=====================\n"
subprocess_command_message += "= End of stdout\n"
subprocess_command_message += "=====================\n\n\n"

subprocess_command_message += "=====================\n"
subprocess_command_message += "= Below is stderr\n"
subprocess_command_message += "=====================\n"
subprocess_command_message += "\n{stderr}\n\n"
subprocess_command_message += "=====================\n"
subprocess_command_message += "= End of stderr\n"
subprocess_command_message += "=====================\n"


def get_path_to_scheduler_template_file():
    cwd = os.getcwd()
    path_to_file = os.path.join(cwd,"lsf.template.x")
    return path_to_file

def get_path_to_snap_cpu_executable(path_to_workspace):
    exe_path = os.path.join(path_to_workspace,"build_directory/SNAP/CPU/snap")
    return exe_path

def get_path_to_snap_gpu_executable(path_to_workspace):
    exe_path = os.path.join(path_to_workspace,"build_directory/SNAP/GPU/snap")
    return exe_path

def get_dirname_of_optjob_snap_inputfiles(path_to_workspace):
    dirname = os.path.join(path_to_workspace,"build_directory/SNAP/opt_jobs/inputs")
    return dirname

def get_dirname_of_basejob_snap_inputfiles(path_to_workspace):
    dirname = os.path.join(path_to_workspace,"build_directory/SNAP/base_jobs/inputs")
    return dirname