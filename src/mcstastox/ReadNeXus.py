import h5py
import os
import numpy as np
import re

# All settings should have a default
defaults = dict(component_numbers=None, # number of digits in component numbers in file
                nd_geometry_info=False, # True when geometry info included
                )

# Version settings, ordered from newest to oldest
mcstas_version_settings = {
    (3, 5, 20): dict(defaults, component_numbers=4, nd_geometry_info=True),
    (2, 7, 0): defaults,
}

class McStasNeXus:
    """
    Reads a McStas NeXus files and provides methods to retrieve data or entries
    """
    def __init__(self, file_handle):
        self.file_handle = file_handle
        f = self.file_handle

        self.mcstas_version = self.read_mcstas_version()

        # Load settings appropriate for this McStas version
        self.settings = None
        for version, settings in mcstas_version_settings.items():
            if self.mcstas_version >= version:
                self.settings = settings
                break

        if self.settings is None:
            raise ValueError("McStas version ", self.mcstas_version, " not supported by this tool.")

        # Check file is formatted as expected
        if "entry1" not in list(f.keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'entry1'.")

        if "data" not in list(f["entry1"].keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'data'.")

        if "simulation" not in list(f["entry1"].keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'simulation'.")

        if "Param" not in list(f["entry1"]["simulation"].keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'Param'.")

        if "instrument" not in list(f["entry1"].keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'instrument'.")

        if "components" not in list(f["entry1"]["instrument"].keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'components'.")

        # Grab basic information
        if self.settings["component_numbers"] is None:
            self.component_names = f["entry1"]["instrument"]["components"].keys()
            self.component_path_names = {name: name for name in self.component_names}
        else:
            comp_name_start_index = self.settings["component_numbers"] + 1
            self.component_names = []
            self.component_path_names = {}
            full_comp_names = f["entry1"]["instrument"]["components"].keys()
            for name in full_comp_names:
                component_name = name[comp_name_start_index:]
                self.component_names.append(component_name)
                self.component_path_names[component_name] = name

    def read_mcstas_version(self):
        f = self.file_handle

        # First attempt at reading version
        if "entry1" not in list(f.keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'entry1'")

        if "simulation" not in list(f["entry1"].keys()):
            raise ValueError("h5 file not formatted as expected, lacks 'entry1/simulation'")

        if "program" not in list(f["entry1"]["simulation"].attrs):
            raise ValueError("h5 file not formatted as expected, lacks 'entry1/simulation/program'")

        version_string = f["entry1"]["simulation"].attrs["program"].decode("utf-8")

        match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_string)
        if match:
            return tuple(map(int, match.groups()))

        # Can have other methods here for older / newer formats

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
            raise ValueError(f"No component with name '{component_name}' found in file.")

        component_name = self.component_path_names[component_name]
        return self.file_handle["entry1"]["instrument"]["components"][component_name]

    def get_geometry_entry(self, component_name):
        """
        :return: the geometry entry of the specified component
        """
        component_entry = self.get_component_entry(component_name)

        if not self.settings["nd_geometry_info"]:
            raise ValueError("The version of McStas used to write this NeXus file did not embed monitor_nD geometry info")

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
            raise ValueError(f"Expected to find {name} in BINS entry of component '{component_name}'")

        axis = np.asarray(bins_entry[name])

        return loaded_var, axis

    def get_x_var_and_axis(self, component_name):
        """
        :return: tuple with loaded x variable, x axis for given component name
        """
        return self.get_var_and_axis(component_name=component_name, var="xvar", label="xlabel")

    def get_y_var_and_axis(self, component_name):
        """
        :return: tuple with loaded y variable, y axis for given component name
        """
        return self.get_var_and_axis(component_name=component_name, var="yvar", label="ylabel")

    def get_z_var_and_axis(self, component_name):
        """
        :return: tuple with loaded z variable, z axis for given component name
        """
        return self.get_var_and_axis(component_name=component_name, var="zvar", label="zlabel")

    def get_geometry_dict(self, component_name):
        """
        Creates a dictionary describing the geometry of given component name

        This dictionary is not directly the format in the NeXus file, but
        is supposed to stay fixed even when the NeXus file may change. This
        method can be updated to reflect these changes and underlying code
        can remain untouched.
        """

        # Method and amount of information depend on McStas version
        if self.settings["nd_geometry_info"]:
            # Use geometry info
            geometry_info = self.get_geometry_entry(component_name)

            # Possible field names, doesn't all need to be there
            # Dict is used in case of name changes on nexus file
            field_names = ["height", "radius", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "Shape identifier"]
            fields = {name : name for name in field_names}

            geometry = {}
            for geometry_name, nexus_field in fields.items():
                if nexus_field in geometry_info.attrs:
                    read_value = geometry_info.attrs[nexus_field].decode("utf-8")

                    # Convert from string to numbers when possible
                    try:
                        read_value = float(read_value)
                    except:
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

            shape_identifier_dict = {0:"square",
                                     1:"disk",
                                     2:"sphere",
                                     3:"cylinder",
                                     4:"banana",
                                     5:"box",
                                     6:"previous",
                                     7:"off"}

            #convert shape to a string using lookup table
            if "Shape identifier" in geometry:
                read_shape_identifier = abs(int(geometry["Shape identifier"]))
                if read_shape_identifier in shape_identifier_dict:
                    geometry["shape"] = shape_identifier_dict[read_shape_identifier]

            return geometry

        else:
            # Deduct as much as possible, only square can really work

            info_entry = self.get_info_entry(component_name)

            if "options" not in info_entry.attrs:
                raise ValueError(f"Expected 'options' in {component_name}, but wasn't found.")

            options = info_entry.attrs["options"].decode("utf-8")

            if "square" in options:
                bins_entry = self.get_BINS_entry(component_name)

                xvar = bins_entry.attrs["xvar"].decode("utf-8")
                yvar = bins_entry.attrs["yvar"].decode("utf-8")

                if xvar.strip() == "x" and yvar.strip() == "y":

                    if "xylimits" not in info_entry.attrs:
                        raise ValueError(f"xylimits exected in NeXus entry for component '{component_name}'")

                    xylimits = info_entry.attrs["xylimits"].decode("utf-8")
                    # Matches floats and integers, including negative and positive numbers
                    matches = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', xylimits)
                    xmin = float(matches[0])
                    xmax = float(matches[1])
                    ymin = float(matches[2])
                    ymax = float(matches[3])
                    xwidth = xmax - xmin
                    yheight = ymax - ymin
                    return dict(shape="square", xwidth=xwidth, yheight=yheight,
                                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

                else:
                    raise ValueError(f"Can't find sufficient info for geometry in component '{component_name}'")

            else:
                raise ValueError("Did not find sufficient information to read geometry, recreate file with newer McStas version")

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
        assert len(contents) == 1

        return output_entry[contents[0]]

    def get_component_n_events(self, component_name):
        """
        :return: get number of events in component with event data
        """

        info_entry = self.get_info_entry(component_name)

        if "events" not in info_entry.keys():
            raise ValueError(f"The component '{component_name}' does not have events entry.")

        return info_entry["events"].shape[0]

    def get_component_events_array(self, component_name):
        """
        :return: get event array from component with event data
        """

        info_entry = self.get_info_entry(component_name)

        if "events" not in info_entry.keys():
            raise ValueError(f"The component '{component_name}' does not have events entry.")

        return np.asarray(info_entry["events"])

    def get_component_parameter_entry(self, component_name):
        """
        :return: returns the component parameter entry of given component name
        """
        component_entry  = self.get_component_entry(component_name)

        if "parameters" not in component_entry.keys():
            raise ValueError(f"The component '{component_name}' does not have a parameter entry")

        return component_entry["parameters"]

    def get_component_parameter_names(self, component_name):
        """
        :return: returns the component parameter names of given component name
        """
        parameter_entry = self.get_component_parameter_entry(component_name)

        return list(parameter_entry.keys())

    def get_component_parameters(self, component_name):
        """
        :return: returns dictionary with parameter names and values of given component name
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
                except:
                    pass

                par_dict[par_name]["value"] = value

            if "default" in this_par_entry.attrs:
                default = this_par_entry.attrs["default"].decode("utf-8")
                try:
                    default = float(default)
                except:
                    pass

                par_dict[par_name]["default"] = default

        return par_dict

    def get_component_variables(self, component_name):
        """
        :return: variables recorded for each event in given component name
        """

        info_entry = self.get_info_entry(component_name)

        if "variables" not in info_entry.attrs:
            raise ValueError(f"The component '{component_name}' does not have variables attribute in info entry.")

        return info_entry.attrs["variables"].decode("utf-8")

    def get_variable_index(self, component_name, variable):
        """
        :return: gets variables index for given variable name for given component name
        """
        variables = self.get_component_variables(component_name)
        return variables.split(" ").index(variable)

    def get_event_data(self, variables, component_name=None):
        """
        :return: event data of given list of variables for given component name (list of names allowed)
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
                    raise ValueError(f"Component {comp} did not have variable {var} in event data")

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