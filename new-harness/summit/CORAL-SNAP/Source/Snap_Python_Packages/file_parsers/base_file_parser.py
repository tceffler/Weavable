#! /usr/bin/env python3
import string

#
# Author: Arnold Tharrington (arnoldt@ornl.gov)
# National Center for Computational Sciences, Scientific Computing Group.
# Oak Ridge National Laboratory
#

class BaseFileParser:

    comment_line_entry = "#"
    separator = "="

    def __init__(self,
                 inputfilename):
        self.__inputFileName = inputfilename


        #
        # Read the input file.
        #
        self.__read_file()

    def __read_file(self):
        ifile_obj = open(self.__inputFileName,"r")
        lines = ifile_obj.readlines()
        ifile_obj.close()
        
        for tmpline in lines:

            # Split the lines into words.
            words = str.split(tmpline)

            # If there are no words, then continue to next line.
            if self.__is_line_empty(words):
                continue

            # If this is a comment line, the continue to next line.
            if self.__is_comment_line(words[0]):
                continue

            # Convert the first word to lower case.
            attribute = str.lower(words[0])
            value = str(words[2])

            # Add the new attribute.
            self.__addNewAttribute(attribute,value)

    def __is_comment_line(self,word):
        if word[0] == self.comment_line_entry:
            return True
        else:
            return False

    def __is_line_empty(self,words):
            if len(words) == 0:
                return True
            else:
                return False

    def __addNewAttribute(self,attribute,value):
        setattr(self,attribute,value)
        return
