# A deep learning approach to estimate stress distribution: a fast and accurate surrogate of finite-element analysis
This repository provides the code for this article:
https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0844

This repository is not the original one published by the authors: 
https://github.com/TML-Gatech/DL_Stress_TAA/tree/master

Instead this repository is written all in Python which is not the case for the original repository (Matlab + Python). The old tenserflow pipeline was replaced by up to date Pytorch versions.


# Dependencies
Main dependencies that are not included in the repo and should be installed first (with PIP or conda):

Pytorch

# Instructions

After cloning the repository, unzip the **ShapeData** and **StressData** in the same directory.  

In your command prompt (Anaconda command prompt in case using Anaconda) activate the environment including the Pytorch, then:

```
python main.py --i 5
```
The argument passed is the number of iterations siwithing the train/test indices in each training. Refer to the article for more information.
