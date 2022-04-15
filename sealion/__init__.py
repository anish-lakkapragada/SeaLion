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

import os
import sys
from pathlib import Path
from subprocess import Popen, DEVNULL

__version__ = "4.4.3"

ROOT = Path(__file__).parent
VERSION_PATH = ROOT / "cython_ran.txt"

PYTHON = sys.executable


def read_cython():
    with open(VERSION_PATH.as_posix(), "r") as fp:
        return fp.read().strip()

def write_cython(data):
    with open(VERSION_PATH.as_posix(), "w") as fp:
        fp.write(data)


def compile_cython():
    args = [PYTHON, (ROOT / "setup.py").as_posix(), "build_ext", "--inplace"]
    return Popen(args, cwd=ROOT.as_posix(), stdout=DEVNULL, stderr=DEVNULL)

def check_cython():
    if not VERSION_PATH.is_file() or read_cython() != __version__:
        # need to recompile cython
        #print("Compiling cython. Please wait...")
        proc = compile_cython()
        proc.wait()

        if proc.returncode != 0:
            raise ValueError(f"SeaLion: Cython compile failed with exit code {proc.returncode}.")

        write_cython(__version__)


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
del Popen
del DEVNULL
del Path

del ROOT
del VERSION_PATH

del read_cython
del write_cython
del compile_cython
del check_cython
