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

__version__ = "4.4.5"

# Cython extensions are built at install time (see setup.py) and are
# imported lazily by the modules that use them, e.g.
# sealion/decision_trees.py imports cython_decision_tree_functions
# inside the method that needs it. Importing sealion itself must
# never compile anything: no subprocess, no build_ext, and no
# cython_ran.txt marker file. If an extension is missing, the
# method that needs it raises ImportError; rebuild with
# `pip install .` or `python setup.py build_ext --inplace`.

from . import regression  # passed
from . import decision_trees
from . import DimensionalityReduction
from . import ensemble_learning
from . import naive_bayes
from . import nearest_neighbors
from . import unsupervised_clustering
from . import utils
from . import neural_networks
