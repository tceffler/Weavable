#! /usr/bin/env python3

#
# Author: Arnold Tharrington
# Email: arnoldt@ornl.gov
# National Center of Computational Science, Scientifc Computing Group.
#

# System imports

# My local imports.
from file_parsers.base_file_parser import BaseFileParser

class snap_configuration_file(BaseFileParser):

    # Constructor method
    def __init__(self,
                 file_name="snap_configuration.txt"):
        super().__init__(file_name)
        return


    #-----------------------------------------------------
    # Public methods                                     -
    #                                                    -
    #-----------------------------------------------------
    @property
    def number_of_atoms_tag(self):
        return str(self.natoms_tag)

    @property
    def batch_queue(self):
        return str(self.queue)

    @property
    def number_of_nodes(self):
        return int(self.nm_of_nodes)

    @property
    def number_of_resource_sets_per_node(self):
        return int(self.nm_resource_set_per_node)

    @property
    def number_of_gpus_per_node(self):
        return int(self.nm_gpus_per_node)

    @property
    def number_of_mpi_tasks_per_node(self):
        return int(self.nm_mpi_tasks_per_node)

    @property
    def number_of_mpi_tasks_per_resource_set(self):
        return int(self.nm_mpi_tasks_per_resource_set) 

    @property
    def number_of_gpus_per_mpi_task(self):
        return int(self.nm_gpus_per_mpi_task)

    @property
    def job_walltime(self):
        return str(self.walltime)

    @property
    def name_of_batchfile(self):
        return str(self.batch_file_name)

    @property
    def number_of_copies(self):
        return int(self.nm_of_copies)

    @property
    def name_of_jobname(self):
        return str(self.jobname)

    #-----------------------------------------------------
    # Private methods                                    -
    #                                                    -
    #-----------------------------------------------------

def main():
    pass

if __name__ == "__main__":
    main()
