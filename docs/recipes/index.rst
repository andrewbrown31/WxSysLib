.. _recipes:

=======
Recipes
=======

This section provides an overview of all of recipes within WxSysLib. Recipes draw from dependent libraries and utility functions to diagnose and track meteorological phenomena.

.. toctree::
   :maxdepth: 2

`Tropical Cyclones <https://github.com/21centuryweather/WxSysLib/blob/main/recipes/tropical_cyclones>`_ 
=======================================================================================================

`Tropical Cyclone Tracking in ERA5 (TempestExtremes) <https://github.com/21centuryweather/WxSysLib/blob/main/recipes/tropical_cyclones/tropical_cyclone_era5.ipynb>`_ 
____________________________________________________________________________________________________________________________________________________________________

The recipe detects and track tropical cyclones using TempestExtremes. Tropical cyclones are identified from mean sea level pressure (MSLP) minima with compact MSLP contours and warm-core structure.

`Rossby Wave Breaking <https://github.com/21centuryweather/WxSysLib/blob/main/recipes/rossby_wave_breaking>`_ 
==============================================================================================================

`Rossby Wave Breaking (by overturned contours) in ERA5 <https://github.com/21centuryweather/WxSysLib/blob/main/recipes/rossby_wave_breaking/detect_and_track_RWB_ERA5.ipynb>`_ 
____________________________________________________________________________________________________________________________________________________________________

This recipe identifies and tracks overturning contours as a proxy for Rossby wave breaking. It smooths and identifies potential temperature overturned contours on the dynamical tropopause (2 PVU surface) at 5K intervals and tracks them by looking for zones which overlap. It can easily be tweaked to track potential vorticity on isentropic surfaces as in Barnes et. al. (2025)
