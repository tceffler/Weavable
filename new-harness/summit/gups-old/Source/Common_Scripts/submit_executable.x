#!/usr/bin/env python
"""
-------------------------------------------------------------------------------
File:   submit_executable.x
Author: Arnold Tharrington (arnoldt@ornl.gov)
Modified: Veronica G. Vergara Larrea, Wayne Joubert
National Center for Computational Sciences, Scientific Computing Group.
Oak Ridge National Laboratory
Copyright (C) 2017 Oak Ridge National Laboratory, UT-Battelle, LLC.
-------------------------------------------------------------------------------
"""

import os
import argparse
import re
import time
from subprocess import Popen, PIPE
import shlex

#------------------------------------------------------------------------------

def process_command_line_args():
    """Get the command line arguments."""

    command_description = (
        'A driver program that the submits the binary thru batch for '
        'testing.  The submit program also writes the job id of the '
        'submitted batch job to the file Status/<test_id_string>/job_id.txt. '
        'The only line in job_id.txt is the job id.')

    p_help = (
        'The absolute path to the workspace. This path must have the '
        'appropiate permissions to permit the user of the test to r, w, and x.')

    i_help = (
        'The test id string. The build program uses this string to make a '
        'unique directory. We don\'t want concurrent builds to clobber each '
        'other.  The submit program uses this string to write the job '
        'schedule id to Status/<test_id_string>/job_id.txt')

    r_help = (
        'The batch script will resubmit itself, otherwise only 1 instance '
        'will be submitted')

    parser = argparse.ArgumentParser(description=command_description)
    parser.add_argument('-p', help=p_help, required=True)
    parser.add_argument('-i', help=i_help, required=True)
    parser.add_argument('-r', help=r_help, action='store_const', const=True)

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------

def main():
    """Main program for submit operation.  Creates a batch script for
       performing a run of the application and submits it to the scheduler.
    """

    # Get the command line arguments.

    args = process_command_line_args()
    workspace_dir = args.p
    test_id = args.i
    batch_recursive_mode = '0' if args.r else '1'

    # Determine which scheduler to use.

    # Make the batch script.

    batch_file_path = make_batch_file(batch_recursive_mode, workspace_dir,
                                      test_id)

    # Submit the batch file to the scheduler.

    sched_job_id = send_to_scheduler(batch_file_path)
    print('submit_executable.x: Job id = ' + str(sched_job_id))

    # Write scheduler job id to job_id.txt in the Status dir.

    write_job_id_to_status(sched_job_id, test_id)

#------------------------------------------------------------------------------

def get_app_test(workspace_dir):
    """Helper: obtain the app name and test name."""

    dir_head1, test_id = os.path.split(workspace_dir)
    dir_head2, test = os.path.split(dir_head1)
    dir_head3, app = os.path.split(dir_head2)

    return app, test

#------------------------------------------------------------------------------

def get_results_dir(test_id):
    """Helper: get path to results dir for test_id in Run_archive dir."""

    dir_head1, dir_tail1 = os.path.split(scripts_dir())
    path = os.path.join(dir_head1, 'Run_Archive', test_id)

    return path

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

def make_batch_file(batch_recursive_mode, workspace_dir, test_id):
    """Create the batch script file to send to scheduler."""

    template_filename = 'lsf.template.x'

    app, test = get_app_test(workspace_dir)

    batch_file_name = ('batchscript_' + app + '_' + test + '.sh')

    # Define the parse definitons and the regular expressions.

    program = 'gups.summit'
    nccstestharnessmodule = os.environ['RGT_NCCS_TEST_HARNESS_MODULE']
    rgtenvironmentalfile = os.environ['RGT_ENVIRONMENTAL_FILE']
    jobname = app + '_' + test
    nodes = num_nodes(test)
    ranks_per_node = '6'
    ranks = str(int(nodes) * int(ranks_per_node))
    batchqueue = (os.environ['RGT_SUBMIT_QUEUE']
                  if 'RGT_SUBMIT_QUEUE' in os.environ else 'batch')
    pbsaccountid = os.environ['RGT_PBS_JOB_ACCNT_ID']
    executable_path = os.path.join(workspace_dir, 'build_directory', 'bin',
                                   program)
    starting_dir = scripts_dir()
    results_dir = get_results_dir(test_id)
    #os.mkdir(results_dir)
    work_dir = os.path.join(workspace_dir, 'workdir')
    resubmitme = batch_recursive_mode
    walltime = '10'

    # Option for delaying (re)submission of job.
    jobdelay_minutes = 0
    timenow = time.time()
    starttime_obj = time.localtime(timenow + (jobdelay_minutes * 60))

    batch_file_header = ''
    starttime = time.strftime('%Y:%m:%d:%H:%M', starttime_obj)
    if jobdelay_minutes > 0:
        batch_file_header += '#BSUB -b %s\n' % (starttime)
    if 'RGT_SUBMIT_ARGS' in os.environ and os.environ['RGT_SUBMIT_ARGS'] != '':
        batch_file_header += '#BSUB %s\n' % (os.environ['RGT_SUBMIT_ARGS'])

    #num_threads = 2 * 10 if is_summitdev() else 2 * 22
    #execution_command = ('env OMP_NUM_THREADS=' + str(num_threads) + ' ' +
    #                     'mpirun --bind-to none -np 1 ' +
    #                     './set_device_and_bind.sh ./' + program + ' ' +
    #                     '1> std.out.txt 2> std.err.txt')

    execution_command = (
                         'jsrun ' +
                         ' --nrs ' + ranks +
                         ' --rs_per_host ' + ranks_per_node +
                         ' --bind rs --cpu_per_rs 7 -g 1' +
                         ' --smpiargs none' +
                         ' --stdio_mode individual ' + 
                         ' --stdio_stdout %h.%t.out --stdio_stderr %h.%t.err' +
                         ' $EXECUTABLE_PATH ' +
                         ' -d 1 -W 2 -s 32 -m R -g S -W 1 -R 5 ' +
                         ' -t M -u 8 -b 8589934592 '
                         ' 1> std.out.txt 2> std.err.txt')

    regex_list = [
        ('__jobname__', jobname),
        ('__walltime__', walltime),
        ('__nodes__', nodes),
        ('__ranks__', ranks),
        ('__ranks_per_node__', ranks_per_node),
        ('__requiredmodules__', nccstestharnessmodule),
        ('__rgtenvironmentalfile__', rgtenvironmentalfile),
        ('__batchqueue__', batchqueue),
        ('__pbsaccountid__', pbsaccountid),
        ('__executable_path__', executable_path),
        ('__starting_dir__', starting_dir),
        ('__results_dir__', results_dir),
        ('__workspace_dir__', workspace_dir),
        ('__work_dir__', work_dir),
        ('__execution_command__', execution_command),
        ('__resubmitme__', resubmitme),
        ('__test_id__', test_id),
        ('__batch_file_name__', batch_file_name),
        ('__starttime__', starttime),
        ('__batch_file_header__', batch_file_header),
        ('__program__', program),
   ]

    # Read the lines of the batch template file.

    template_file = open(template_filename, 'r')
    lines = template_file.readlines()
    template_file.close()

    # Make the batch file from the template.

    #batch_file_path = batch_file_name
    batch_file_path = os.path.join(workspace_dir, batch_file_name)

    batch_file = open(batch_file_path, 'w')
    for line in lines:
        for regexp, repltext in regex_list:
            line = re.sub(regexp, repltext, line)
        batch_file.write(line)
    batch_file.close()

    return batch_file_path

#------------------------------------------------------------------------------

def write_job_id_to_status(sched_job_id, test_id):
    """Write scheduler job id to job_id.txt in the Status dir."""

    # Get path to file.

    dir_head1, dir_tail1 = os.path.split(scripts_dir())
    path1 = os.path.join(dir_head1, 'Status', test_id, 'job_id.txt')

    # Write the pbs job id to the file.

    fileobj = open(path1, 'w')
    string1 = '%20s\n' % (sched_job_id)
    fileobj.write(string1)
    fileobj.close()

    return path1

#------------------------------------------------------------------------------

def send_to_scheduler(batch_file_path):
    """Submit batch script to scheduler."""

    #print('submit_executable.x: using LSF scheduler syntax to submit job')

    # Set the appropriate queueing command for each scheduler

    submit_command = 'bsub '
    #qcommand = submit_command
    qcommand = submit_command + batch_file_path

    # Split the arguments for the command

    args = shlex.split(qcommand)

    # Execute the command as a subprocess

    my_jobfile = open(batch_file_path, 'r')
    #process = Popen(args, stdout=PIPE, stderr=PIPE, stdin=my_jobfile)
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    my_jobfile.close()

    output, err = process.communicate()

    print(output.decode('utf-8').split('\n'))
    print(err.decode('utf-8').split('\n'))

    records = output.decode('utf-8').split('\n')
    jobid = extract_jobid(records)

    return jobid

#------------------------------------------------------------------------------

def extract_jobid(records):
    """Extract the scheduler job id from the output string."""

    #print('submit_executable.x: extracting LSF jobID')
    #jobid = re.compile(r'\d+').findall(records[0])[0]
    jobid = ''
    if len(records) > 0:
        matches = re.compile(r'\d+').findall(records[0])
        if len(matches) > 0:
            jobid = matches[0]
    if jobid == '':
        print('Unable to obtain job id.')

    return jobid

#------------------------------------------------------------------------------

def scripts_dir():
    """The Scripts/ directory."""
    return os.getcwd()

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------
