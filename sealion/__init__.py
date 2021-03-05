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

from . import regression  # passed
from . import decision_trees
from . import DimensionalityReduction
from . import ensemble_learning
from . import naive_bayes
from . import nearest_neighbors
from . import unsupervised_clustering
from . import utils
from . import neural_networks


import time
import os
import subprocess
import pickle
import shutil
import sys

start = time.time()

org_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

VERSION_NUMBER = "4.2.8"
PYTHON_RUNNING_VERSION = str(sys.version_info.major) + "." + str(sys.version_info.minor)


def read_pickle_file(file) :
    with open(file, 'rb') as f :
        stuff = pickle.load(f)
    return stuff

if os.path.exists("cython_ran.pickle") :
    if read_pickle_file("cython_ran.pickle") == VERSION_NUMBER :
        # then we are up to date here
        pass
    else :
        # this means that it was a different version
        var = subprocess.Popen(
            f"python{PYTHON_RUNNING_VERSION} setup.py build_ext --inplace",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        while True:
            # files are currently being generated
            try:
                from . import cython_decision_tree_functions
                from . import cython_knn
                from . import cython_naive_bayes
                from . import cython_unsupervised_clustering
                from . import cython_tsne
                from . import cython_ensemble_learning
                from . import cython_mixtures
                break
            except Exception:
                if time.time() - start > 500:
                    print(
                        "Cython compilation files unable to load. Please raise this error on github, "
                        "and especially include your architecture, OS, and Python version. Thank you!"
                    )

                    break

        with open("cython_ran.pickle", "wb") as f:
            pickle.dump(VERSION_NUMBER, f) # save the version number
else:
    var = subprocess.Popen(
        f"python{PYTHON_RUNNING_VERSION} setup.py build_ext --inplace",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    while True:
        # files are currently being generated
        try:
            from . import cython_decision_tree_functions
            from . import cython_knn
            from . import cython_naive_bayes
            from . import cython_unsupervised_clustering
            from . import cython_tsne
            from . import cython_ensemble_learning
            from . import cython_mixtures
            break
        except Exception:
            if time.time() - start > 500:
                print(
                    "Cython compilation files unable to load. Please raise this error on github, "
                    "and especially include your architecture, OS, and Python version. Thank you!"
                )

                break

    with open("cython_ran.pickle", "wb") as f:
        pickle.dump(VERSION_NUMBER, f)


os.chdir(org_dir)
