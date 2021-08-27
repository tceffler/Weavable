#! /usr/bin/env python3

#
# Author: Arnold Tharrington
# Email: arnoldt@ornl.gov
# National Center of Computational Science, Scientifc Computing Group.
#

# System imports
import os
import sys
import io
import re
import shutil
import copy
import subprocess
import shlex
from datetime import datetime

# Local imports
import build_utilities
import job_submitter_exceptions

class JobFeatures:
    def __init__(self):
        return

class JobSubmitter:
    def __init__(self):
        self.__initializationStatus = False
        self.__templateRecords = None
        self.__newBacthFileRecords = None
        self.__listOfFeatures = None
        self.__features = None
        self.__batchfilename = None
        self.__jobid = None
        return

    #-----------------------------------------------------
    # Public methods                                     -
    #                                                    -
    #-----------------------------------------------------
    @property
    def InitializationStatus(self):
        return self.__initializationStatus

    @InitializationStatus.setter
    def InitializationStatus(self,value):
        self.__initializationStatus = value
        return

    @property
    def BatchFileName(self):
        return self.__batchfilename

    @property
    def JOBID(self):
        return self.__jobid

    @JOBID.setter
    def JOBID(self,value):
        self.__jobid = value

    def submitJob(self,
                  mode="Normal"):
        submit_scheduler_return_code = None
        if self.InitializationStatus and mode=="Normal":
            self.formNewJobSubmissionScript()
            submit_scheduler_return_code = self.submitToScheduler()
        elif self.InitializationStatus and mode=="Debug":
            self.formNewJobSubmissionScript()
        else:
            submit_scheduler_return_code = build_utilities.unix_exit_status["failure"]
            message =  "Not submitting a job.\n"
            message += "The job submitter is not properly initialized.\n"
            print(message)
            return submit_scheduler_return_code 
        return submit_scheduler_return_code 

    def initialize(self,
                   batchfilename=None):

        self.__batchfilename = batchfilename
        self.__templateRecords = self.readTemplateRecords()
        self.__newBacthFileRecords = []
        self.__features = JobFeatures
        self.__listOfFeatures = []
        self.InitializationStatus = True
        return

    def addFeature(self,name,value,rgexp):
        setattr(self.__features,name,(rgexp,value))
        self.__listOfFeatures += [name]
        my_attribute = getattr(self.__features,name)
        message = "Added attribue {_name_} = {_rgexp_} ; {_value_}.\n\n\n".format(_name_=name,
                                                                      _rgexp_=my_attribute[0],
                                                                      _value_=my_attribute[1])
        return

    def getSchedulerTemplate(self):
        file_path = build_utilities.get_path_to_scheduler_template_file()
        return file_path

    def formNewJobSubmissionScript(self):
        fileobj = open(self.__batchfilename,"w")

        try:
            for record1 in self.__templateRecords:
                record2 = copy.deepcopy(record1)
                for name_of_feature in self.__listOfFeatures:
                    (my_regexp,value) = getattr(self.__features,name_of_feature)
                    if not isinstance(value,str) :
                        raise job_submitter_exceptions.JobSubmitterTypeError(value,record2)
                    record2 = my_regexp.sub(value,record2)
                fileobj.write(record2)
        except job_submitter_exceptions.JobSubmitterTypeError as my_exception:
            my_exception.what()
            fileobj.close()
            sys.exit(3)

        fileobj.close()
        return

    #-----------------------------------------------------
    # Private methods                                    -
    #                                                    -
    #-----------------------------------------------------
    def __changeMyStatusToInitialized(self):
        self.__initializationStatus = True

    def __changeMyStatusToNotInitialized(self):
        self.__initializationStatus = False

    #-----------------------------------------------------
    # Static methods                                     -
    #                                                    -
    #-----------------------------------------------------



class PeakJobSubmitter(JobSubmitter):
    def __init__(self,
                 type=None):
        super().__init__()
        self.__type = type
        self.__submitCommand = 'bsub "-env all" '
        self.__logFilePath = "lammps.bsub.logfile.txt"
        self.__logfileFobj = None
        return
    
    #-----------------------------------------------------
    # Public methods                                     -
    #                                                    -
    #-----------------------------------------------------
    @property
    def LOGFILE_FILEOBJ(self):
        return self.__logfileFobj

    @LOGFILE_FILEOBJ.setter
    def LOGFILE_FILEOBJ(self,value):
        self.__logfileFobj = value

    @property
    def LOGFILE_PATH(self):
        return self.__logFilePath 

    def submitToScheduler(self):
        lsf_submit_command = self.__submitCommand

        my_launcher_command = "{job_launcher_command} {batch_file_name}".format(
                job_launcher_command = lsf_submit_command,
                batch_file_name = self.BatchFileName)

        my_launcher_command_2 = "{job_launcher_command} {batch_file_name} 1> lsf.stdout 2> lsf.stderr".format(
                job_launcher_command = lsf_submit_command,
                batch_file_name = self.BatchFileName)

        my_bsub_process = None

        with open(self.LOGFILE_PATH,"w") as self.LOGFILE_FILEOBJ:

            build_message = "{currenttime}: Initiated bsub for lammps.\n".format(currenttime=datetime.now())

            args = shlex.split(lsf_submit_command)
            my_bsub_process = subprocess.run(my_launcher_command,
                                             shell=True,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)

            message = build_message 
            message += build_utilities.subprocess_command_message.format(execdir=os.getcwd(),
                                                         returncode = my_bsub_process.returncode,
                                                         command=my_launcher_command,
                                                         stdout=my_bsub_process.stdout.decode('utf-8'),
                                                         stderr=my_bsub_process.stderr.decode('utf-8'))
            self.LOGFILE_FILEOBJ.write(message)

            self.parseStringForJobId(my_bsub_process.stdout.decode('utf-8'))

        return my_bsub_process.returncode

    def readTemplateRecords(self):
        my_scheduler_template = self.getSchedulerTemplate() 

        templatefileobject = open(my_scheduler_template,"r")
        tlines = templatefileobject.readlines()
        templatefileobject.close()

        return tlines

    def parseStringForJobId(self,
                            my_string):

        words = my_string.split()
        job_id = words[1].replace("<","")
        job_id = job_id.replace(">","")
        self.JOBID = job_id
        return

    #-----------------------------------------------------
    # Private methods                                    -
    #                                                    -
    #-----------------------------------------------------

    #-----------------------------------------------------
    # Static methods                                     -
    #                                                    -
    #-----------------------------------------------------

SummitJobSubmitter = PeakJobSubmitter

    
