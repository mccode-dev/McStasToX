# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import h5py
import numpy as np


@dataclass(frozen=True)
class McStasVersionSetting:
    component_numbers: int | None = None
    """Number of digits in component numbers in file"""
    nd_geometry_info: bool = False
    """True when geometry info included"""


# McStas version settings registry.
# Easy cheat-sheet for each version of McStas.
# Keep it rrdered from the newest to the oldest for easier maintenance.
_McStasVersionSettingTp = MappingProxyType[tuple[int, int, int], McStasVersionSetting]
_MCSTAS_VERSION_SETTINGS: _McStasVersionSettingTp = MappingProxyType(
    {
        (3, 5, 20): McStasVersionSetting(component_numbers=4, nd_geometry_info=True),
        (2, 7, 0): McStasVersionSetting(),  # Use default values between 2.7 and 3.5.20
    }
)


def _get_mcstas_version_settings(
    version: tuple[int, int, int],
    mcstas_version_setting_registry: _McStasVersionSettingTp = _MCSTAS_VERSION_SETTINGS,
) -> McStasVersionSetting:
    """
    Get the settings for a given McStas version.
    """
    latest_version_first_sorted_settings = sorted(
        mcstas_version_setting_registry.items(), key=lambda x: x[0], reverse=True
    )
    for ver, settings in latest_version_first_sorted_settings:
        if version >= ver:
            return settings
    raise ValueError(f"McStas version {version} not supported by this tool.")


def _validate_file(file_handle: h5py.File) -> None:
    # Check file is formatted as expected
    if "entry1" not in list(file_handle.keys()):
        raise ValueError("h5 file not formatted as expected, lacks 'entry1'.")

    entry_obj = cast(h5py.Group, file_handle["entry1"])
    entry_keys = tuple(entry_obj.keys())

    mandatory_entry_keys = ("data", "simulation", "instrument")
    missing_keys = tuple(key for key in mandatory_entry_keys if key not in entry_keys)
    if any(missing_keys):
        raise ValueError(
            "'entry1' not formatted as expected, lacks keys: "
            f"[{', '.join(missing_keys)}]."
        ) from None

    simluation_keys = tuple(entry_obj["simulation"].keys())
    if "Param" not in simluation_keys:
        raise ValueError(
            "'entry1/simulation' not formatted as expected, lacks 'Param'."
        )

    instrument_keys = tuple(entry_obj["instrument"].keys())
    if "components" not in instrument_keys:
        raise ValueError(
            "'entry1/instrument' not formatted as expected, lacks 'components'."
        )


class McStasNeXus:
    """
    Reads a McStas NeXus files and provides methods to retrieve data or entries

    """

    def __init__(
        self,
        file_handle,
        *,
        mcstas_version: tuple[int, int, int] | None = None,
        mcstas_setting_registry: _McStasVersionSettingTp = _MCSTAS_VERSION_SETTINGS,
    ):
        self.file_handle = file_handle
        # Check file is formatted as expected
        _validate_file(self.file_handle)
        self.mcstas_version = mcstas_version or self._read_mcstas_version()

        # Load settings appropriate for this McStas version
        self.settings = _get_mcstas_version_settings(
            self.mcstas_version, mcstas_setting_registry
        )

        # Grab basic information
        self.component_names: list
        self.component_path_names: dict
        self._read_component_name_and_path()

    def _read_component_name_and_path(self) -> None:
        if self.settings.component_numbers is None:
            self.component_names = list(
                self.file_handle["entry1"]["instrument"]["components"].keys()
            )
            self.component_path_names = {name: name for name in self.component_names}
        else:
            comp_name_start_index = self.settings.component_numbers + 1
            components = self.file_handle["entry1"]["instrument"]["components"].keys()
            component_paths = list(components)
            self.component_names = [
                name[comp_name_start_index:] for name in component_paths
            ]
            self.component_path_names = dict(
                zip(self.component_names, component_paths, strict=True)
            )

    def _read_mcstas_version(self) -> tuple[int, int, int]:
        f = self.file_handle
        if "program" not in list(f["entry1"]["simulation"].attrs):
            raise ValueError(
                "h5 file not formatted as expected, "
                "lacks 'program' attribute in 'entry1/simulation/program'"
            )

        version_string = f["entry1"]["simulation"].attrs["program"].decode("utf-8")

        match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_string)
        if match:
            return tuple(map(int, match.groups()))

        # Can have other methods here for older / newer formats
        # If not, raise error
        raise ValueError(
            f"Could not find version in file, found '{version_string}."
            "Hardcode the version as an argument to the class."
        )

    def get_components(self):
        """
        :return: list of component names
        """
        return list(self.component_names)

    def get_components_with_data(self):
        """
        :return: list of component names that have data
        """
        components_with_data = []

        for comp in self.component_names:
            component_entry = self.get_component_entry(comp)
            if "output" in list(component_entry.keys()):
                components_with_data.append(comp)

        return list(components_with_data)

    def get_components_with_ids(self):
        """
        :return: list of component names that have pixel id's
        """
        components_with_ids = []
        for comp in self.get_components_with_data():
            output_entry = self.get_output_entry(comp)
            output_contents = list(output_entry.keys())

            if "BINS" in output_entry and len(output_contents) > 1:
                # Need both BINS entry and data output to have data
                components_with_ids.append(comp)

        return list(components_with_ids)

    def get_components_with_geometry(self):
        """
        :return: list of component names that has geometry info
        """
        components_with_geometry = []
        for comp in self.get_components_with_data():
            comp_entry = self.get_component_entry(comp)
            if "Geometry" in comp_entry:
                components_with_geometry.append(comp)

        return list(components_with_geometry)

    def get_component_entry(self, component_name):
        """
        :return: the component entry of the specified component
        """
        if component_name not in self.component_path_names:
            raise ValueError(
                f"No component with name '{component_name}' found in file."
            )

        component_name = self.component_path_names[component_name]
        return self.file_handle["entry1"]["instrument"]["components"][component_name]

    def get_geometry_entry(self, component_name):
        """
        :return: the geometry entry of the specified component
        """
        component_entry = self.get_component_entry(component_name)

        if not self.settings.nd_geometry_info:
            raise ValueError(
                "The version of McStas used to write this NeXus file "
                "did not embed monitor_nD geometry info"
            )

        if "Geometry" not in list(component_entry.keys()):
            raise ValueError(f"'{component_name}' does not have geometry data.")

        return component_entry["Geometry"]

    def get_output_entry(self, component_name):
        """
        :return: the output entry of the specified component
        """
        component_entry = self.get_component_entry(component_name)

        if "output" not in list(component_entry.keys()):
            raise ValueError(f"'{component_name}' does not have data.")

        return component_entry["output"]

    def get_BINS_entry(self, component_name):
        """
        :return: the BINS entry of the specified component
        """
        output_entry = self.get_output_entry(component_name)

        if "BINS" not in output_entry.keys():
            raise ValueError(f"Component {component_name} does not have BINS entry")

        return output_entry["BINS"]

    def get_var_and_axis(self, component_name, var, label):
        """
        :param component_name: str:  string with component name
        :param var: str:  variable name, for example "xvar"
        :param label: str: label name, for example "xlabel"
        :return: tuple with loaded variable, axis
        """
        bins_entry = self.get_BINS_entry(component_name)

        if var not in bins_entry.attrs:
            return None, None

        loaded_var = bins_entry.attrs[var].decode("utf-8")
        loaded_label = bins_entry.attrs[label].decode("utf-8")
        # Replace special characters with underscore
        name = re.sub(r'[^a-zA-Z]', '_', loaded_label)

        if name not in bins_entry:
            raise ValueError(
                f"Expected to find {name} in BINS entry of component '{component_name}'"
            )

        axis = np.asarray(bins_entry[name])

        return loaded_var, axis

    def get_x_var_and_axis(self, component_name):
        """
        :return: tuple with loaded x variable, x axis for given component name
        """
        return self.get_var_and_axis(
            component_name=component_name, var="xvar", label="xlabel"
        )

    def get_y_var_and_axis(self, component_name):
        """
        :return: tuple with loaded y variable, y axis for given component name
        """
        return self.get_var_and_axis(
            component_name=component_name, var="yvar", label="ylabel"
        )

    def get_z_var_and_axis(self, component_name):
        """
        :return: tuple with loaded z variable, z axis for given component name
        """
        return self.get_var_and_axis(
            component_name=component_name, var="zvar", label="zlabel"
        )

    def get_geometry_dict(self, component_name):
        """
        Creates a dictionary describing the geometry of given component name

        This dictionary is not directly the format in the NeXus file, but
        is supposed to stay fixed even when the NeXus file may change. This
        method can be updated to reflect these changes and underlying code
        can remain untouched.
        """

        # Method and amount of information depend on McStas version
        if self.settings.nd_geometry_info:
            # Use geometry info
            geometry_info = self.get_geometry_entry(component_name)

            # Possible field names, doesn't all need to be there
            # Dict is used in case of name changes on nexus file
            field_names = [
                "height",
                "radius",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "zmin",
                "zmax",
                "Shape identifier",
            ]
            fields = {name: name for name in field_names}

            geometry = {}
            for geometry_name, nexus_field in fields.items():
                if nexus_field in geometry_info.attrs:
                    read_value = geometry_info.attrs[nexus_field].decode("utf-8")

                    # Convert from string to numbers when possible
                    try:
                        read_value = float(read_value)
                    except TypeError:
                        pass

                    geometry[geometry_name] = read_value

            if "xmin" in geometry and "xmax" in geometry:
                geometry["xwidth"] = geometry["xmax"] - geometry["xmin"]

            if "ymin" in geometry and "ymax" in geometry:
                geometry["yheight"] = geometry["ymax"] - geometry["ymin"]

            if "zmin" in geometry and "zmax" in geometry:
                geometry["zdepth"] = geometry["zmax"] - geometry["zmin"]

            """
            # Code from monitor-nd-lib.c encoding the shape
            DEFS->SHAPE_SQUARE =0;    /* shape of the monitor */
            DEFS->SHAPE_DISK   =1;
            DEFS->SHAPE_SPHERE =2;
            DEFS->SHAPE_CYLIND =3;
            DEFS->SHAPE_BANANA =4;
            DEFS->SHAPE_BOX    =5;
            DEFS->SHAPE_PREVIOUS=6;
            DEFS->SHAPE_OFF=7;
            """

            shape_identifier_dict = {
                0: "square",
                1: "disk",
                2: "sphere",
                3: "cylinder",
                4: "banana",
                5: "box",
                6: "previous",
                7: "off",
            }

            # convert shape to a string using lookup table
            if "Shape identifier" in geometry:
                read_shape_identifier = abs(int(geometry["Shape identifier"]))
                if read_shape_identifier in shape_identifier_dict:
                    geometry["shape"] = shape_identifier_dict[read_shape_identifier]

            return geometry

        else:
            # Deduct as much as possible, only square can really work

            info_entry = self.get_info_entry(component_name)

            if "options" not in info_entry.attrs:
                raise ValueError(
                    f"Expected 'options' in {component_name}, but wasn't found."
                )

            options = info_entry.attrs["options"].decode("utf-8")

            if "square" in options:
                bins_entry = self.get_BINS_entry(component_name)

                xvar = bins_entry.attrs["xvar"].decode("utf-8")
                yvar = bins_entry.attrs["yvar"].decode("utf-8")

                if xvar.strip() == "x" and yvar.strip() == "y":
                    if "xylimits" not in info_entry.attrs:
                        raise ValueError(
                            "xylimits eceeded in NeXus entry "
                            f"for component '{component_name}'"
                        )

                    xylimits = info_entry.attrs["xylimits"].decode("utf-8")
                    # Matches floats and integers,
                    # including negative and positive numbers
                    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', xylimits)
                    xmin = float(matches[0])
                    xmax = float(matches[1])
                    ymin = float(matches[2])
                    ymax = float(matches[3])
                    xwidth = xmax - xmin
                    yheight = ymax - ymin
                    return dict(
                        shape="square",
                        xwidth=xwidth,
                        yheight=yheight,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                    )

                else:
                    raise ValueError(
                        "Can't find sufficient info for "
                        f"geometry in component '{component_name}'"
                    )

            else:
                raise ValueError(
                    "Did not find sufficient information to read geometry, "
                    "recreate file with newer McStas version"
                )

    def get_pixels_entry(self, component_name):
        """
        :return: pixel entry of given component name
        """
        bins_entry = self.get_BINS_entry(component_name)

        if "pixels" not in bins_entry.keys():
            raise ValueError("This component does not a pixels entry.")

        return bins_entry["pixels"]

    def get_info_entry(self, component_name):
        """
        :return: info entry of given component name
        """
        output_entry = self.get_output_entry(component_name)

        # Get data, there may be BINS and a data entry with a weird name
        contents = list(output_entry.keys())
        if "BINS" in contents:
            contents.remove("BINS")

        # Ensure there is only one element
        if len(contents) != 1:
            raise AssertionError(
                f"Expected only one entry from '{component_name}'"
                f" but found {len(contents)} entries."
            )

        return output_entry[contents[0]]

    def get_component_n_events(self, component_name):
        """
        :return: get number of events in component with event data
        """

        info_entry = self.get_info_entry(component_name)

        if "events" not in info_entry.keys():
            raise ValueError(
                f"The component '{component_name}' does not have events entry."
            )

        return info_entry["events"].shape[0]

    def get_component_events_array(self, component_name):
        """
        :return: get event array from component with event data
        """

        info_entry = self.get_info_entry(component_name)

        if "events" not in info_entry.keys():
            raise ValueError(
                f"The component '{component_name}' does not have events entry."
            )

        return np.asarray(info_entry["events"])

    def get_component_parameter_entry(self, component_name):
        """
        :return: returns the component parameter entry of given component name
        """
        component_entry = self.get_component_entry(component_name)

        if "parameters" not in component_entry.keys():
            raise ValueError(
                f"The component '{component_name}' does not have a parameter entry"
            )

        return component_entry["parameters"]

    def get_component_parameter_names(self, component_name):
        """
        :return: returns the component parameter names of given component name
        """
        parameter_entry = self.get_component_parameter_entry(component_name)

        return list(parameter_entry.keys())

    def get_component_parameters(self, component_name):
        """
        :return: Dictionary with parameter names
                 and values of given component name
        """

        par_entry = self.get_component_parameter_entry(component_name)

        par_dict = {}
        par_names = self.get_component_parameter_names(component_name)

        for par_name in par_names:
            par_dict[par_name] = {}
            this_par_entry = par_entry[par_name]

            if "type" in this_par_entry.attrs:
                par_type = this_par_entry.attrs["type"].decode("utf-8")
                par_dict[par_name]["type"] = par_type

            if "value" in this_par_entry.attrs:
                value = this_par_entry.attrs["value"].decode("utf-8")
                try:
                    value = float(value)
                except TypeError:
                    pass

                par_dict[par_name]["value"] = value

            if "default" in this_par_entry.attrs:
                default = this_par_entry.attrs["default"].decode("utf-8")
                try:
                    default = float(default)
                except TypeError:
                    pass

                par_dict[par_name]["default"] = default

        return par_dict

    def get_component_variables(self, component_name):
        """
        :return: variables recorded for each event in given component name
        """

        info_entry = self.get_info_entry(component_name)

        if "variables" not in info_entry.attrs:
            raise ValueError(
                f"The component '{component_name}' does not "
                "have variables attribute in info entry."
            )

        return info_entry.attrs["variables"].decode("utf-8")

    def get_variable_index(self, component_name, variable):
        """
        :return: gets variables index for given variable name for given component name
        """
        variables = self.get_component_variables(component_name)
        return variables.split(" ").index(variable)

    def get_event_data(self, variables, component_name=None):
        """
        :return: event data of given list of variables
                 for given component name (list of names allowed)
        """

        if component_name is None:
            # Default is to gather data for all components with pixel id's
            components_with_ids = self.get_components_with_ids()
        else:
            # Allow component_name to be a list of names, convert if it is not
            if not isinstance(component_name, list):
                component_name = [component_name]

            components_with_ids = component_name

        # Get total length of return arrays first
        total_length = 0
        ranges = {}
        for comp in components_with_ids:
            ranges[comp] = dict(start=total_length)
            total_length += self.get_component_n_events(comp)
            ranges[comp]["end"] = total_length

        # Check variables contained in all components
        for comp in components_with_ids:
            comp_variables = self.get_component_variables(comp)
            for var in variables:
                if var not in comp_variables:
                    raise ValueError(
                        f"Component {comp} did not have variable {var} in event data"
                    )

        # Allocate return arrays
        returns = {}
        for var in variables:
            returns[var] = np.empty(total_length)

        # Fill return arrays with requested data
        for comp in components_with_ids:
            array = self.get_component_events_array(comp)
            start = ranges[comp]["start"]
            end = ranges[comp]["end"]
            for var in variables:
                var_index = self.get_variable_index(comp, var)
                returns[var][start:end] = array[:, var_index]

        return returns
