#! /usr/bin/env python3

########################################################################
# Author: Arnold Tharrington                                           #
# Email: arnoldt@ornl.gov                                              #
# National Center of Computational Science, Scientifc Computing Group. #
########################################################################

class JobSubmitterError(Exception):
    def __init__(self):
        super().__init__()
        return

class JobSubmitterTypeError(JobSubmitterError):
    ###################
    # Class variables #
    ###################

    ###################
    # Special methods #
    ###################

    def __init__(self,
                 name,
                 value):
        super().__init__()
        self.__name = name
        self.__value = value
        return

    ###################
    # Public methods  #
    ###################

    def what(self):
        message =  "The JobSubmitter has encounterd a type coercion.\n"
        message += "Trying to coerce {} of type {} to a string type.".format(
                    self.__name,
                    type(self.__name))
        print(message)
        return


def main():
    pass

if __name__ == "main":
    main()
