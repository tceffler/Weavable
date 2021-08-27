#! /usr/bin/env python
"""
-------------------------------------------------------------------------------
File:   check_executable.x
Author: Arnold Tharrington (arnoldt@ornl.gov)
Modified: Veronica G. Vergara Larrea, Wayne Joubert
National Center for Computational Sciences, Scientific Computing Group.
Oak Ridge National Laboratory
Copyright (C) 2017 Oak Ridge National Laboratory, UT-Battelle, LLC.
-------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import re
import fnmatch

#------------------------------------------------------------------------------

IS_PASSING_YES = 1
IS_PASSING_NO = 0

#------------------------------------------------------------------------------

def process_command_line_args():
    """Get the command line arguments."""

    command_description = (
        'A program that checks the results located at <path_to_results>. '
        'The check executable must write the status of the results to the '
        'file Status/<test_id_string>/job_status.txt')

    p_help = 'The absoulte path to the results of a test.'

    i_help = 'The test id string.'

    parser = argparse.ArgumentParser(description=command_description)
    parser.add_argument('-p', help=p_help, required=True)
    parser.add_argument('-i', help=i_help, required=True)

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------

def main():
    """Main program for check operation.  Check the correctness of
       the run results and report back.
    """

    # Get the command line arguments.

    args = process_command_line_args()
    path_to_results = args.p
    #test_id = args.i

    # Compare the results.

    is_passing = check_results(path_to_results)

    # Write the status of the results to job data file.

    write_to_job_status_file(path_to_results, is_passing)

#------------------------------------------------------------------------------

def get_test(path_to_results):
    """Extract test name from path to results. """

    dir_head1, test_id = os.path.split(path_to_results)
    dir_head2, run_archive = os.path.split(dir_head1)
    dir_head3, test = os.path.split(dir_head2)

    return test

#------------------------------------------------------------------------------

def is_summitdev():
    """Are we running on Summitdev (as opposed to Summit). """
    return len(re.findall(r'summitdev', os.environ['HOSTNAME'])) > 0

#------------------------------------------------------------------------------

def num_nodes(test):
    """Extract number of nodes to run on from test name."""

    q = r'?' # Use this to avoid vim syntax coloring bug.
    matches = re.findall(r'^.*' + q + r'_([0-9]+)node_.*', '_' + test + '_')
    assert len(matches) == 1
    match = matches[0]

    return re.sub('^0*', '', match)

#------------------------------------------------------------------------------

def check_results(path_to_results):
    """Perform the correctness check of the results."""

    perf_targets = {
        r'AGG' : 725,
    }

    test = get_test(path_to_results)

    nodes = num_nodes(test)
    ranks_per_node = '6'
    ranks = int(nodes) * int (ranks_per_node)

    num_passed = 0

    # Iterate over output files, checking each as we go

    file_list = os.listdir(path_to_results)
    outfile_pattern = "*.*.out"
    for entry in file_list:  
        if fnmatch.fnmatch(entry, outfile_pattern):
            file_path = os.path.join(path_to_results, entry)
            file_ = open(file_path, 'r')
            lines = file_.readlines()
            file_.close()
            for line in lines:
                tokens = re.split(' +', re.sub(r':', '', line.rstrip()))
                if len(tokens) != 15:
                    continue
                measure = tokens[12]
                if measure in perf_targets:
                    value = float(tokens[13])
                    target = float(perf_targets[measure])
                    print(entry, measure, value, target)
                    if value < target:
                        print('check_executable error: ' + entry +
                              ' missed performance target for ' + measure +
                              ', was ' + str(value) + ', expected ' + str(target))
                    else:
                        num_passed += 1

    num_total = 1 * ranks
    if num_passed != num_total:
        print('check_executable error: ' + str(num_passed) + ' of ' +
              str(num_total) + ' performance checks passed.')
        return IS_PASSING_NO

    print('Correctness check passed successfully.')

    return IS_PASSING_YES

#------------------------------------------------------------------------------

def write_to_job_status_file(path_to_results, is_passing):
    """Write the status of the results to job data file."""

    # Get path.

    dir_head1, dir_tail1 = os.path.split(path_to_results)
    dir_head2, dir_tail2 = os.path.split(dir_head1)
    file_path = os.path.join(dir_head2, 'Status', dir_tail1, 'job_status.txt')

    file_ = open(file_path, 'w')

    # Create the the string to write.

    if is_passing == IS_PASSING_NO:
        indicator = '1'
    elif is_passing == IS_PASSING_YES:
        indicator = '0'
    elif is_passing >= 2:
        indicator = '2'
    string_ = '%s\n' % (indicator)

    # Write the string.

    file_.write(string_)
    file_.close()

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------
