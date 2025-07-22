Contribution Guidelines
=======================
When contributing to WxSysLib, the following guidelines should be adhered to. 

Installed packages
------------------
The code within WxSysLib are python-based fucntions and tools which are largely built on top of well-established libraries and tools in a variety of languages. Any package dependencies should be clearly noted in any documentation and function descriptions. For Australian and NCI users, these packages are installed locally on NCI. The libaries should be installed onto the NCI WxSysLib  and should be added, in consultation with the NCI project owners and the WxSysLib working group.


Utility functions
-----------------
Utility functions are python fucntions within WxSysLib that can be reused in a number of different applications and recipes. Codes contirbutions, wherever possible, should be written as a series of utility functions. Utility functions are to be located in the utils directory. The utils directory is sorted into the various category directories. New contributions shoudl be divided into these categories. The util subdirectories include:

- blobs - for all threshold identification and tracking functions (e.g. TempestExtremes blob utilities, TOBAC utilities). 
- nodes - for all point identification and tracking functions (e.g. TempestExtremes node utilities)
- lines - for all line identification and tracking functions
- diagnostics - all instantaneous dynamical diagnostic funcitons (e.g. Lagranto utilities)
- general - all general utilities that can be used across all fucntions (e.g. datetime.datetime iterators, nci specific functions)
- plotting - all generalised plotting fucntions that can be used for a range of applciations (e.g. synoptic chart plots, point density climatologies)

Recipes
-------


Documentation
-------------



