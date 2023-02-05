# AST
Python code for classification of environmental sounds. 


# Data
Environmental sound files are contained in the folder audio in .wav format.

# Preprocessing
The preprocessing.py file contains the code for filtering the signals anb computing the spectra using the Mel filter bank.
Requiremets: 
import pandas as pd
import numpy as np
import scipy.io.wavfile
import scipy.signal as sps 

# Patching and labeling 
The Patching.py file contains the code for computing 16X16 patches from the original images (spectra) and its corresponding vectorization. 
Labels are also computed, and patches and labels are saved in the folder Patches.
Requiremets: 
import numpy as np
import pandas as pd

# Documentation
The codes included in this repository where created following the guidelines in the papers included in this folder
