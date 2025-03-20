import h5py
import os
import numpy as np
import re

from .ReadNeXus import McStasNeXus


class Data:
    def __init__(self, data_folder, filename="mccode.h5"):
        # Open the file and store the file object as an instance attribute
        self.file = h5py.File(os.path.join(data_folder, filename), "r")
        self.file_object = McStasNeXus(self.file)

        # Prepare data structure for when data is requested
        # List for component names in sequence of lowest to highest pixel ID
        self.component_pixel_order = []

        # Dictionaries with keys of component names
        self.pixel_range = {}  # list of len 2, lowest and highest pixel ID
        self.local_pixel_locations = {}  # list of length
        self.global_pixel_locations = {}

    def close(self):
        # Close the file when done
        if self.file:
            self.file.close()

    # Enable context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_components(self):
        return list(self.file_object.component_names)

    def get_components_with_data(self):
        return self.file_object.get_components_with_data()

    def get_components_with_ids(self):
        return self.file_object.get_components_with_ids()

    def get_components_with_geometry(self):
        return self.file_object.get_components_with_geometry()

    def show_components(self):
        print("All components in file:")
        comps = self.get_components()
        for comp in comps:
            print(" ", comp)

    def show_components_with_data(self):
        comps = self.get_components_with_data()
        if len(comps) == 0:
            print("No components with data in file:")
        else:
            print("All components with data in file:")
            for comp in comps:
                print(" ", comp)

    def show_components_with_ids(self):
        comps = self.get_components_with_ids()
        if len(comps) == 0:
            print("No components with pixel id information in file:")
        else:
            print("All components with pixel id information in file:")
            for comp in comps:
                print(" ", comp)

    def show_components_with_geometry(self):
        comps = self.get_components_with_geometry()
        if len(comps) == 0:
            print("No components with geometry information in file:")
        else:
            print("All components with geometry information in file:")
            for comp in comps:
                print(" ", comp)

    def get_event_data(self, variables, component_name=None, filter_zeros=True):

        event_data = self.file_object.get_event_data(variables=variables, component_name=component_name)
        if "p" in variables and filter_zeros:
            # Remove 0 events
            zero_weights = np.where(event_data["p"] == 0)[0]
            for key in event_data:
                event_data[key] = np.delete(event_data[key], zero_weights)

        return event_data

    def get_component_variables(self, component_name):
        return self.file_object.get_component_variables(component_name)

    def get_component_placement(self, component_name):
        component_entry = self.file_object.get_component_entry(component_name)

        return np.asarray(component_entry["Position"]), np.asarray(component_entry["Rotation"])

    def get_global_component_coordinates(self, component_name):
        return self.transform(np.zeros((1, 3)), component_name)[0]

    def get_component_data(self, component_name):
        data = self.file_object.get_output_entry(component_name)

        if "events" in data.keys():
            return np.asarray(data["events"])
        elif "data" in data.keys() and "errors" in data.keys() and "ncount" in data.keys():
            I = np.asarray(data["data"])
            E = np.asarray(data["errors"])
            N = np.asarray(data["ncount"])
            return I, E, N

    def calculate_pixel_locations_direct(self, component_name):

        bins_entry = self.file_object.get_BINS_entry(component_name)
        info_entry = self.file_object.get_info_entry(component_name)
        pixels = self.file_object.get_pixels_entry(component_name)

        xvar = bins_entry.attrs["xvar"]
        xlabel = bins_entry.attrs["xlabel"].decode("utf-8")
        # Replace special characters with underscore
        x_name = re.sub(r'[^a-zA-Z]', '_', xlabel)

        if x_name not in bins_entry:
            raise ValueError(f"Expected to find {x_name} in BINS entry")

        yvar = bins_entry.attrs["yvar"]
        ylabel = bins_entry.attrs["ylabel"].decode("utf-8")
        y_name = re.sub(r'[^a-zA-Z]', '_', ylabel)

        if y_name not in bins_entry:
            raise ValueError(f"Exected to find {y_name} in BINS entry")

        x_axis = np.asarray(bins_entry[x_name])
        y_axis = np.asarray(bins_entry[y_name])

        pixels = np.asarray(pixels, dtype="int")

        # Find meshgrid
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)

        if "options" not in info_entry.attrs:
            raise ValueError(f"Expected 'options' in {component_name}, but wasn't found.")

        options = info_entry.attrs["options"].decode("utf-8")

        if "square" in options:
            local_x = x_grid.ravel()
            local_y = y_grid.ravel()
            local_z = np.zeros(len(local_x))

        elif "banana" in options:
            radius = 1
            print("Warning, set radius of banana detector to 1 as radius not available in NeXus file")
            theta = x_grid.ravel() * np.pi / 180

            local_x = radius * np.sin(theta)
            local_y = y_grid.ravel()
            local_z = radius * np.cos(theta)
        else:
            raise ValueError("Unknown geometry")

        coordinates = np.column_stack((local_x, local_y, local_z))
        self.store_and_transform(coordinates, pixels, component_name)

    def calculate_pixel_locations(self, component_name):

        xvar, x_axis = self.file_object.get_x_var_and_axis(component_name)
        yvar, y_axis = self.file_object.get_y_var_and_axis(component_name)
        zvar, z_axis = self.file_object.get_z_var_and_axis(component_name)

        pixels = self.file_object.get_pixels_entry(component_name)
        pixels = np.asarray(pixels, dtype="int")

        geometry = self.file_object.get_geometry_dict(component_name)

        if geometry["shape"] == "square":
            if not (xvar == "x" and yvar == "y"):
                raise ValueError(f"The monitor {component_name} does not record both x and y for pixel positions")

            x_grid, y_grid = np.meshgrid(x_axis, y_axis)
            local_x = x_grid.ravel()
            local_y = y_grid.ravel()
            local_z = np.zeros(len(local_x))

        elif geometry["shape"] == "banana":
            if not (xvar == "th" and yvar == "y"):
                raise ValueError(f"The monitor {component_name} does not record both theta and y for pixel positions")

            x_grid, y_grid = np.meshgrid(x_axis, y_axis)
            radius = geometry["radius"]

            theta = x_grid.ravel() * np.pi / 180

            local_x = radius * np.sin(theta)
            local_y = y_grid.ravel()
            local_z = radius * np.cos(theta)

            ###### Can add the remaining monitor nd shapes with their transformations
        else:
            raise ValueError("Unknown geometry")

        coordinates = np.column_stack((local_x, local_y, local_z))
        self.store_and_transform(coordinates, pixels, component_name)

    def store_and_transform(self, coordinates, pixels, component_name):
        self.local_pixel_locations[component_name] = coordinates

        min_pixel = np.min(pixels)
        max_pixel = np.max(pixels)
        self.pixel_range[component_name] = [min_pixel, max_pixel]

        # Check for overlap
        for comp in self.component_pixel_order:
            comp_min_pixel = self.pixel_range[comp][0]
            comp_max_pixel = self.pixel_range[comp][1]

            if comp_min_pixel <= min_pixel <= comp_max_pixel or comp_min_pixel <= max_pixel <= comp_max_pixel:
                raise ValueError(f"Overlap of pixel id's between {comp} and {component_name}")

        # Find point in sequence
        if len(self.component_pixel_order) == 0:
            self.component_pixel_order.append(component_name)
        else:
            # Quick check to see if it should be added to end
            last_comp = self.component_pixel_order[-1]
            last_comp_max = self.pixel_range[last_comp][1]
            if min_pixel > last_comp_max:
                self.component_pixel_order.append(component_name)
            else:
                for index, comp in enumerate(self.component_pixel_order):
                    comp_min_pixel = self.pixel_range[comp][0]
                    if min_pixel < comp_min_pixel:
                        self.component_pixel_order.insert(index, component_name)
                        break

        self.global_pixel_locations[component_name] = self.transform(coordinates, component_name)

    def transform(self, coordinates, component_name):
        pos, rot = self.get_component_placement(component_name)
        return coordinates @ rot + pos

    def get_component_global(self, component_name):

        if component_name not in self.global_pixel_locations:
            self.calculate_pixel_locations(component_name)

        return self.global_pixel_locations[component_name]

    def get_component_local(self, component_name):

        if component_name not in self.local_pixel_locations:
            self.calculate_pixel_locations(component_name)

        return self.local_pixel_locations[component_name]

    def load_all_with_id(self):

        id_components = self.get_components_with_ids()
        for comp in id_components:
            if comp not in self.global_pixel_locations:
                self.calculate_pixel_locations(comp)

    def check_id_continuous(self):
        # Load all monitors that have pixel id's
        self.load_all_with_id()

        # Only possible if there is nice continuous coverage of pixel id's
        last_end = -1
        for comp in self.component_pixel_order:
            comp_min_pixel = self.pixel_range[comp][0]
            comp_max_pixel = self.pixel_range[comp][1]
            if comp_min_pixel != last_end + 1:
                return False

            last_end = comp_max_pixel

        return True

    def get_highest_id(self):
        # Load all monitors that have pixel id's
        self.load_all_with_id()
        # The above call fails if there are pixel ID overlaps, but allows gaps

        # Provide the largest ID, last component end of range
        last_comp = self.component_pixel_order[-1]
        return self.pixel_range[last_comp][1]

    def get_id_to_coordinate(self, component_name=None, local=False):
        # Load all monitors that have pixel id's
        self.load_all_with_id()

        # Length of array should be the highest ID + 1
        length = self.get_highest_id() + 1

        if component_name is None:
            # Default is to gather data for all components with pixel id's
            components = self.component_pixel_order
        else:
            # Allow component_name to be a list of names, convert if it is not
            if not isinstance(component_name, list):
                components = [component_name]
            else:
                components = component_name

        result = np.ma.masked_all((length, 3), dtype=np.float64)
        for comp in components:
            if local:
                coordinates = self.local_pixel_locations[comp]
            else:
                coordinates = self.global_pixel_locations[comp]

            comp_min_pixel = self.pixel_range[comp][0]
            comp_max_pixel = self.pixel_range[comp][1]
            result[comp_min_pixel:comp_max_pixel + 1, :] = coordinates

        return result

    def get_id_to_global_coordinates(self, component_name=None):
        return self.get_id_to_coordinate(local=False, component_name=component_name)

    def get_id_to_local_coordinates(self, component_name=None):
        return self.get_id_to_coordinate(local=True, component_name=component_name)

    def export_scipp_simple(self, source_name, sample_name, component_name=None,
                            filter_zeros=True, extra_variables=None):
        try:
            import scipp as sc
        except:
            raise ImportError("Scipp installation required to export to Scipp format")

        variables = ["p", "t", "id"]
        if extra_variables is not None:
            if not isinstance(extra_variables, list):
                extra_variables = [extra_variables]
            # Default is to gather weight, time and id
            variables += extra_variables

        event_data = self.get_event_data(variables=variables, component_name=component_name,
                                         filter_zeros=filter_zeros)

        # Retrieve coordinates corresponding to id's
        global_coordinates = self.get_id_to_global_coordinates(component_name=component_name)
        global_pos = global_coordinates[event_data["id"].astype(int), :]

        source_pos = self.get_global_component_coordinates(source_name)
        sample_pos = self.get_global_component_coordinates(sample_name)

        events = sc.DataArray(
            data=sc.array(dims=['events'], unit=sc.units.counts, values=event_data["p"]),
            coords={
                'position': sc.vectors(dims=['events'], values=global_pos, unit='m'),
                't': sc.array(dims=['events'], unit='s', values=event_data["t"]),
                'source_position': sc.vector(source_pos, unit='m'),
                'sample_position': sc.vector(sample_pos, unit='m'),
            })

        return events


    def export_scipp(self, source_name, sample_name, component_name=None,
                     filter_zeros=True, extra_variables=None):
        try:
            import scipp as sc
        except:
            raise ImportError("Scipp installation required to export to Scipp format")

        # Get the full pixel id range from nexus file
        # Make as generator to work in chunks

        variables = ["p", "t", "id"]
        if extra_variables is not None:
            if not isinstance(extra_variables, list):
                extra_variables = [extra_variables]
            # Default is to gather weight, time and id
            variables += extra_variables

        event_data = self.get_event_data(variables=variables, component_name=component_name,
                                         filter_zeros=filter_zeros)

        # Retrieve coordinates corresponding to id's
        global_coordinates = self.get_id_to_global_coordinates(component_name=component_name)
        #global_pos = global_coordinates[event_data["id"].astype(int), :]

        source_pos = self.get_global_component_coordinates(source_name)
        sample_pos = self.get_global_component_coordinates(sample_name)

        events = sc.DataArray(
            data=sc.array(dims=['events'], unit=sc.units.counts, values=event_data["p"]),
            coords={
                'pixel_id': sc.array(dims=['events'], values=event_data["id"].astype(int)),
                't': sc.array(dims=['events'], unit='s', values=event_data["t"]),
                'source_position': sc.vector(source_pos, unit='m'),
                'sample_position': sc.vector(sample_pos, unit='m'),
            })

        id_object = sc.vectors(dims=['pixel_id'], values=global_coordinates, unit='m')

        lowest_id_comp = self.component_pixel_order[0]
        lowest_id = self.pixel_range[lowest_id_comp][0]
        highest_id_comp = self.component_pixel_order[-1]
        highest_id = self.pixel_range[highest_id_comp][-1]
        #id_range = sc.vector(dims=['pixel_id'], values=[lowest_id, highest_id])

        id_matrix = []
        names = []
        for comp in self.component_pixel_order:
            id_matrix.append(self.pixel_range[comp])
            names.append(comp)

        comp_id_range = sc.array(dims=['panel_id', 'pixel'], values=id_matrix)
        bank_names = sc.array(dims=['panel_id'], values=names)

        output_object = sc.DataGroup(events=events, positions=id_object,
                                     bank_ids=comp_id_range,
                                     bank_names=bank_names)

        output_object["events"] = sc.group(output_object["events"], "pixel_id")
        output_object["events"].coords["position"] = sc.vectors(dims=["pixel_id"],
                                                                values=output_object["positions"].values[output_object["events"].coords["pixel_id"].values, :],
                                                                unit="m")

        return output_object
