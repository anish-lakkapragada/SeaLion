"""
SeaLion is a simple machine learning and data science library.

=============================================================

Designed with beginners in mind, it has rich documentation via Pydoc and algorithms that span from the most basic
to more modern approaches. It is meant to help beginners navigate it all, and the documentation not only explains t
he models and their respective functions but also what they are and when to use them. Emphasis was also put on creating
new functions to make it interesting for those who are just getting started and seasoned ml-engineers alike.

I hope you enjoy it!
- Anish Lakkapragada 2021
"""

import sys
import os
import subprocess

PARENT = os.path.dirname(os.path.realpath(__file__))
CYTHON_RAN_PATH = os.path.join(PARENT, "cython_ran.txt")

VERSION_NUMBER = "4.4.3"


def read_cython():
    with open(CYTHON_RAN_PATH, "r") as fp:
        return fp.read()

def write_cython(data):
    with open(CYTHON_RAN_PATH, "w") as fp:
        fp.write(data)


def compile_cython():
    args = ["python3", os.path.join(PARENT, "setup.py"), "build_ext", "--inplace"]
    return subprocess.Popen(args, cwd=PARENT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def check_cython():
    if not os.path.isfile(CYTHON_RAN_PATH) or read_cython().strip() != VERSION_NUMBER:
        # need to recompile cython
        print("Compiling cython. Please wait...")
        proc = compile_cython()
        proc.wait()

        if not proc.returncode:
            raise ValueError(f"Cython compile failed with exit code {proc.returncode}.")

        write_cython(VERSION_NUMBER)


from . import regression  # passed
from . import decision_trees
from . import DimensionalityReduction
from . import ensemble_learning
from . import naive_bayes
from . import nearest_neighbors
from . import unsupervised_clustering
from . import utils
from . import neural_networks

check_cython()

del sys
del os
del subprocess

del PARENT
del CYTHON_RAN_PATH
del VERSION_NUMBER

del read_cython
del write_cython
del compile_cython
del check_cython
