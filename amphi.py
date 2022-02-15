"""
Author: Satwik Srivastava (^__^)
Profiles =>
    LinkedIn : https://www.linkedin.com/in/satwiksrivastava/
    twitter  : @Satwik_9
    github   : https://github.com/Satwik-S9

- This is a personal project that has some of the most common Machine Learning algorithms implemented into it.
- All the algorithms are implemented from scratch using basic libraries such as numpy, pandas and matplotlib.
- The comments are done using "Better Comments" extension for vs-code see its syntax for ehat each type of comment means.

# todo: Implement identity and diagonal covariance matrices in GMM
# todo: Implement returning labels in GMM
#// todo: Complete plot functions for the models :: completed
"""

# Importing Dependencies
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style 
style.use('ggplot')

from sklearn.datasets import make_spd_matrix
from matplotlib.patches import Ellipse
from scipy import stats



#? Principal Component Analysis


class HMM:
    def __init__(self) -> None:
        self. x = None
