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

1. Python function docstrings

Python documentation strings (or docstrings) provide a convenient way of associating documentation with Python modules, functions, classes, and methods. Docstrings are added to the source code and articulate what the function does, not how.

There are a number of different styles of doctrings. WxSysLib makes use of the `NumpyDoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ docstring style. WxSysLib docstrings should contain:

- A brief description of the function and what it is does
- Parameters required for input, including the data type
- Returned output

2. Recipe comments

Recipes added to WxSysLib should clearly describe and comment on the processes they perform. By default, Recipes are Jupyter notebooks with markdown comments and descriptions, describing each command and process. Recipes should contain:

- A description of the overall recipe, what it does and wwith which methods
- Citations that a user would need to include. This includes original developers of the underlying libraries and tools and the original publication the recipe was developed for. 
- Headline comments of each command and function, what it does and what output may be expected
- A unit test with output that can be compared if run successfully (for example, a plot of tracked cyclones over a month). A good example will highlight and demonstrate all of the features of the recipe. 

3. readthedocs documentation 


