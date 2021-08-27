#! /usr/bin/env python3

# System imports
import os
import re
import abc
from abc import ABC

# Local imports
import build_utilities
from file_parsers.snap_configuration_file import snap_configuration_file

class SnapVerifier(ABC):

    ###################
    # Special methods #
    ###################

    def __init__(self):
        return

    ###################
    # Public methods  #
    ###################
    
    # Abstract methods.
    @abc.abstractmethod
    def validate(self):
        return 

    @abc.abstractmethod
    def getGrindTimes(self):
        return

    def createSnapConfigurationParser(self):
        my_snap_configuration = snap_configuration_file("snap_configuration.txt")
        return my_snap_configuration 

    def parseGrindTimeFromFile(self,
                               filename):
        return [filename,100.00]

#-----------------------------------------------------
# GPU Snap verifier.                                 -
#                                                    -
#-----------------------------------------------------
@SnapVerifier.register
class SnapGPUVerifier(SnapVerifier):

    ###################
    # Special methods #
    ###################

    def __init__(self,
                 path_to_results,
                 test_id_string):

        self.__PathToResults = path_to_results
        self.__TestIDString = test_id_string

        return

    ###################
    # Public methods  #
    ###################

    def validate(self):

        my_snap_configuration = self.createSnapConfigurationParser()

        nm_snap_copies = my_snap_configuration.number_of_copies

        #-----------------------------------------------------
        # Get the grind times of each copy.                  -
        #                                                    -
        #-----------------------------------------------------
        grind_times = self.getGrindTimes(nm_snap_copies)
        
        return build_utilities.verification_results["BAD_RESULTS"]


    def getGrindTimes(self,
                      nm_snap_copies):

        # Form the  unix regular expression to match the snap output files.
        pattern = "out\.gpu\."
        regexp = re.compile(pattern)

        # Form a list of snap output files.
        my_snap_directories1 = [os.path.join(self.__PathToResults,'workdir',"snap_copy_{}".format(ip)) for ip in range(nm_snap_copies)]

        my_snap_directories2 = [ f for f in my_snap_directories1 
                                   if os.path.isdir(f) ]

        my_snap_output_files = [os.path.join(f,y) for f in my_snap_directories2 
                                                  for y in os.listdir(f)
                                                  if  regexp.match(y) ]

        # From the list of output files get the grind times.
        grid_times = [ self.parseGrindTimeFromFile(f) for f in my_snap_output_files 
                                                      if os.path.isfile(f) ]
        return grind_times

#-----------------------------------------------------
# CPU Snap verifier.                                 -
#                                                    -
#-----------------------------------------------------
@SnapVerifier.register
class SnapCPUVerifier(SnapVerifier):

    ###################
    # Special methods #
    ###################
    
    def __init__(self,
                 path_to_results,
                 test_id_string):

        self.__PathToResults = path_to_results
        self.__TestIDString = test_id_string

        return

    ###################
    # Public methods  #
    ###################

    def validate(self):

        my_snap_configuration = self.createSnapConfigurationParser()

        nm_snap_copies = my_snap_configuration.number_of_copies
        
        #-----------------------------------------------------
        # Get the grind times of each copy.                  -
        #                                                    -
        #-----------------------------------------------------
        grind_times = self.getGrindTimes(nm_snap_copies)

        return build_utilities.verification_results["BAD_RESULTS"]

    def getGrindTimes(self,
                      nm_snap_copies):
        return


