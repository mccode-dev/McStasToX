[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![PyPI badge](http://img.shields.io/pypi/v/mcstastox.svg)](https://pypi.python.org/pypi/mcstastox)
[![Anaconda-Server Badge](https://anaconda.org/mccode-dev/mcstastox/badges/version.svg)](https://anaconda.org/mccode-dev/mcstastox)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

# McStasToX

## About

Python package to read McStas data and export as python objects in different formats or as other files.

## DISCLAIMER: This project is still at prototype stage
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

```
pip install mcstastox
```

## Installation
> Note that still an early version of the package is on pip, but the API is not stable yet.

```sh
python -m pip install mcstastox
```

## Basic use
To for example export to scipp, one needs to specify which component is the source and which is the sample.

Then all monitors with pixel id's are loaded, this is purely supported by Monitor_nD with option strings such as:

```
square x bins=512 y bins=256, neutron pixel min=0 t, list all neutrons
square x bins=1024 y bins=512, neutron pixel min=131073 t, list all neutrons
```

The next pixel min should be large enough to accommodate all pixels of the previous monitors, so here above 512*256.

This data can then be loaded to a simple scipp object with:

```
import mcstastox
with mcstastox.Read(file_path) as loaded_data:
    scipp_data = loaded_data.export_scipp_simple(source_name="source", sample_name="sample_position")
```

This data can also  be loaded to a scipp data group with:
```
import mcstastox
with mcstastox.Read(file_path) as loaded_data:
    scipp_data_group = loaded_data.export_scipp(source_name="source", sample_name="sample_position")
```

This takes less space and events are already grouped by pixel ids
