#!/usr/bin/env python
"""
-------------------------------------------------------------------------------
File:   build_executable.x
Author: Wayne Joubert (joubert@ornl.gov)
National Center for Computational Sciences, Scientific Computing Group.
Oak Ridge National Laboratory
Copyright (C) 2017 Oak Ridge National Laboratory, UT-Battelle, LLC.
-------------------------------------------------------------------------------
"""

import os
import argparse
import shutil
import re

#------------------------------------------------------------------------------

def process_command_line_args():
    """Get the command line arguments."""

    command_description = (
        'A driver program that builds the binary for the test.')

    p_help = (
        'The absolute path to the workspace. This path must have the '
        'appropiate permissions to permit the user of the test to r, w, and x.')

    i_help = (
        'The test id string. The build program uses this string to make a '
        'unique directory within path_to_workspace. We don\'t want concurrent '
        'builds to clobber each other.  The submit program uses this string '
        'to write the job schedule id to Status/<test_id_string>/job_id.txt')

    #---The -s flag is not required by the acceptance test harness.
    s_help = 'Path to the source directory (optional).'

    parser = argparse.ArgumentParser(description=command_description)
    parser.add_argument('-p', help=p_help, required=True)
    parser.add_argument('-i', help=i_help, required=True)
    parser.add_argument('-s', help=s_help, required=False)

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------

def main():
    """Main program for building the executable."""

    # Get the command line arguments.

    args = process_command_line_args()
    path_to_workspace = args.p
    #test_id = args.i

    path_to_source = args.s

    # Create the temporary workspace.
    # Save the tempoary workspace for the submit executable.

    # create_tmp_workspace(path_to_workspace)

    # Make the binary.

    build_dir_path = prepare_to_make(path_to_workspace, path_to_source)

    dir_head1, _ = os.path.split(build_dir_path)
    dir_head2, _ = os.path.split(dir_head1)
    _, test = os.path.split(dir_head2)

    exit_status = make_binary(build_dir_path, test)
    if exit_status != 0:
        return 1

    return 0

#------------------------------------------------------------------------------

def prepare_to_make(path_to_workspace, path_to_source):
    """Perform preparations, e.g., copying source tree."""

    if path_to_source is None:

        # Get the current working directory.
        cwd = os.getcwd()

        # Get the 2 tail paths in the cwd.
        dir_head1, dir_tail1 = os.path.split(cwd)
        dir_head2, dir_tail2 = os.path.split(dir_head1)

        # Get the path to the Source directory for the application.
        path_to_source = os.path.join(dir_head2, 'Source')

    # Now make the path to the build directory.

    build_dir_path = os.path.join(path_to_workspace, 'build_directory')

    # Copy Source to build directory.

    print('Copying source tree ...')
    shutil.copytree(path_to_source, build_dir_path, symlinks=True)

    # Change to build directory.

    os.chdir(build_dir_path)

    return build_dir_path

#------------------------------------------------------------------------------

def form_make_args(test):
    """Set up command line arguments for make command."""

    return ''

#------------------------------------------------------------------------------

def make_binary(build_dir_path, test):
    """Execute the make command to build the executable."""

    #make_command = './make.sh ' + form_make_args(test)

    make_command = ('echo "Make not required: '
                          'binary executable already provided."')

    exit_status = os.system(make_command)

    print(build_dir_path)

    return exit_status

#------------------------------------------------------------------------------

def create_tmp_workspace(path1):
    """Create the workspace dir."""
    os.makedirs(path1)

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------
