#! /usr/bin/env python3

import shutil
import os
import sys

# Modify the python path for local submit_executable.py module
cwd = os.getcwd()
sys.path.append(cwd) 

import submit_executable

id="1538407724.9914367"
path_to_workspace="/gpfs/alpinetds/stf006/scratch/arnoldt/Lammps/CORAL-SNAP/Opt_Job/" + id 
batch_recursive_mode=False

submit_executable.submit_executable(path_to_workspace,
                                    id,
                                    batch_recursive_mode)
