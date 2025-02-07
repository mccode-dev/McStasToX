# McStasToX
Python package to read McStas data and export as python objects in different formats or as other files.

## This project is still at prototype stage
The current version of the code is supposed to be a starting point to figure out how such a package should be structured, especially with regards to the API for the user. The demo notebook shows how the package can be used.

### Dependencies
The core package only depends on
- h5py
- numpy

The different export formats, such as scipp, should not be core dependencies but only used when exporting to that format.

To run the demo notebook one will need
- Recent McStas through conda (>3.5.20)
- mcstasscript
- matplotlib
- scipp
- scippneutron
