Contribution Guidelines
========================
When contributing to WxSysLib, the following guidelines should be adhered to. 

Installed packages
------------------
The code within WxSysLib are python-based functions and tools which are largely built on top of well-established libraries and tools in a variety of languages. Any package dependencies should be clearly noted in any documentation and function descriptions. For Australian and NCI users, these packages are installed locally on NCI. Any additional libraries should be installed into the NCI WxSysLib toolbox in consultation with the NCI project owners and the WxSysLib working group. All python library dependencies should be installed in the NCI WxSysLib conda environment. 

Utility functions
------------------
Utility functions are python functions within WxSysLib that can be reused in a number of different applications and recipes. Codes contirbutions, wherever possible, should be written as a series of utility functions. They can have dependencies on packages outside of utils (as described in "Installed packages") but they must be described and installed in the WxSysLib NCI toolbox. Utility functions are to be located in the utils directory. The utils directory is sorted into the various category directories. New contributions shoudl be divided into these categories. The util subdirectories include:

- blobs - for all areal feature identification and tracking functions (e.g. TempestExtremes blob utilities, TOBAC utilities). 
- nodes - for all point identification and tracking functions (e.g. TempestExtremes node utilities)
- lines - for all line identification and tracking functions
- diagnostics - all instantaneous dynamical diagnostic funcitons (e.g. particle tracking utilities)
- general - all general utilities that can be used across all functions (e.g. datetime.datetime iterators, nci specific functions)
- plotting - all generalised plotting functions that can be used for a range of applciations (e.g. synoptic chart plots, point density climatologies)

Recipes
------------------
Recipes are Juypter notebooks within WxSysLib that describe the process for tracking and diagnosis of particular features. Where possible, recipes should be easy to read and rely on a series of utility functions. Recipies are located in the recipies directory and categorised by meteorological phenomenon (for example: tropical_cyclones, extratropical_cyclones, blocking). Recipes that use different techniques to track and diagnose similar phenomena are encouraged! For example, although there is a recipe tracking tropical cyclones using TempestExtremes already in WxSysLib, recipes that use another tool/process to track tropical cyclones are encouraged. 

Documentation
------------------
All submissions should include well-structured documentation of all the functions and recipes. Documentation in WxSysLib is handled in three parts.

1. docstring function documentation

2. In-line recipe commentary documentation

3. readthedocs documentation 


