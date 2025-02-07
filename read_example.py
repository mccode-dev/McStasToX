import mcstasscript as ms
import matplotlib.pyplot as plt

def make_instrument(square=True, banana=True, id_overlap=False, id_gap=False, non_zero_id_start=False):

    instr = ms.McStas_instr("test")
    source = instr.add_component("source", "Source_simple")
    source.set_parameters(xwidth=0.01, yheight=0.01, focus_xw=0.01, focus_yh=0.01, dist=2,
                          lambda0=instr.add_parameter("wavelength", value=2.0), 
                          dlambda=instr.add_parameter("delta_wavelength", value=0.1))

    sample_position = instr.add_component("sample_position", "Arm")
    sample_position.set_AT(source.dist, RELATIVE=source)

    sample = instr.add_component("sample", "PowderN", RELATIVE=sample_position)
    sample.set_parameters(radius=source.xwidth/2, yheight=source.focus_yh, reflections='"Cu.laz"')

    detector_index = 0
    pixel_min = 0

    if non_zero_id_start:
        pixel_min += 1278

    if square:

        detector_direction = instr.add_component("detector_direction_square_1", "Arm", RELATIVE=sample_position, ROTATED=[0, -140, 0])
        
        xbins = 15
        ybins = 15
        monitor = instr.add_component("Square_1", "Monitor_nD")
        monitor.set_parameters(xwidth=0.1, yheight=0.1,
                               filename='"direct_event_square_signal.dat"', restore_neutron=1)
        monitor.options = f'"mantid square x bins={xbins} y bins={ybins}, neutron pixel min={pixel_min} t, l, list all neutrons"'

        monitor.set_AT(0.35, RELATIVE=detector_direction)

        detector_index += 1
        pixel_min += xbins*ybins

        if id_gap:
            pixel_min += 371

        if id_overlap:
            pixel_min -= 52

        detector_direction = instr.add_component("detector_direction_square_2", "Arm", RELATIVE=sample_position, ROTATED=[20, 57, 0])
        
        xbins = 30
        ybins = 15
        monitor = instr.add_component("Square_2", "Monitor_nD")
        monitor.set_parameters(xwidth=0.25, yheight=0.1,
                               filename='"scattered_event_square_signal.dat"', restore_neutron=1)
        monitor.options = f'"mantid square x bins={xbins} y bins={ybins}, neutron pixel min={pixel_min} t, list all neutrons"'

        monitor.set_AT(0.5, RELATIVE=detector_direction)

        detector_index += 1
        pixel_min += xbins*ybins
        
    if banana:
        detector_direction = instr.add_component("detector_direction_banana_1", "Arm", RELATIVE=sample_position, ROTATED=[0, -100, 0])
        
        xbins = 20
        ybins = 12
        monitor = instr.add_component("Banana_1", "Monitor_nD")
        monitor.set_parameters(radius=1.0, yheight=0.1,
                               filename='"direct_event_banana_signal.dat"', restore_neutron=1)
        monitor.options = f'"mantid banana theta bins={xbins} limits=[-10, 25] y bins={ybins}, neutron pixel min={pixel_min} t, list all neutrons"'

        monitor.set_AT(0.0, RELATIVE=detector_direction)

        detector_index += 1
        pixel_min += xbins*ybins

        if id_gap:
            pixel_min += 371

        if id_overlap:
            pixel_min -= 52
        
        detector_direction = instr.add_component("detector_direction_banana_2", "Arm", RELATIVE=sample_position, ROTATED=[0, 120, 0])
        
        xbins = 20
        ybins = 10
        monitor = instr.add_component("Banana_2", "Monitor_nD")
        monitor.set_parameters(radius=1.5, yheight=0.25,
                               filename='"scattered_event_banana_signal.dat"', restore_neutron=1)
        monitor.options = f'"mantid banana theta bins={xbins} limits=[-50, 50] y bins={ybins}, neutron pixel min={pixel_min} t, list all neutrons"'

        monitor.set_AT(0.0, RELATIVE=detector_direction)
        monitor.set_ROTATED([0,0,80], RELATIVE=detector_direction)

        detector_index += 1
        pixel_min += xbins*ybins

    instr.settings(NeXus=True)

    return instr


def plot(array_3d, small_arrays=None, points=None):
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the values
    every = 1
    ax.scatter(array_3d[::every, 2],
               array_3d[::every, 0],
               array_3d[::every, 1],
               c = range(0,len(array_3d[::every,0])), marker='.')

    if small_arrays is not None:
        for small_array in small_arrays:
            ax.scatter(small_array[:, 2],
                       small_array[:, 0],
                       small_array[:, 1],
                       c = 'b', marker='.')

    if points is not None:
        for point in points:
            ax.scatter(point[2],
                       point[0],
                       point[1],
                       c = 'k', marker='.')
        
    ax.set_xlabel('Z-axis')
    ax.set_ylabel('X-axis')
    ax.set_zlabel('Y-axis')
    
    plt.show()