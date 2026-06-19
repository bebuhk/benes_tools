## bene 2026-06-19: copied from /Users/bene/Documents/Projects/epse/master-thesis-local-code/final_server_files/thesis/paper-3d_dft_feos_saft/plot_ExtPot_densities.py

#%% plot_ext-pot&densities.py
# author: Benedikt Buhk
# date: 13.12.2024
# description: plot the external potential and/or densities of a given system (MOF, adsorbate,  + pressure, temperature for densities)

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
# import matplotlib.pyplot as plt
# from matplotlib import cm

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import Normalize
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from matplotlib import cm

#no global variables
## this is LEGACY from my master's thesis
def plot_3d_LJpotential_density(framework_grid, external_pot_grid, temperature, volume_unit_cell_A3, cif:str, adsorbate:str, pressure=None, segment_idx:int = -1, type:str="ExtPot", threshold=None, title=None, subtitle=None, density_info=False, \
                                x=1.25, y=1.25, z=1.25):
    """
    Plot the external potential or densities of a given system in 3D using Plotly.
    Inputs:
    - framework_grid: 4D NumPy array containing the coordinates of the framework grid points
    - external_pot_grid: 3D NumPy array containing the external potential values [kJ/molK] or densities [mol/m^3] at each grid point
    - temperature: temperature of the system [K]
    - pressure: pressure of the system [bar]
    - volume_unit_cell_A3: volume of the unit cell in Å³
    - cif: cif file of the system [eg RSM0016.cif]
    - adsorbate: name of the adsorbate
    - segment_idx: index of the segment to plot (optional)
    - type: type of data to plot, either 'ExtPot' for external potential or 'density' for densities
    - threshold: threshold value to filter data (optional) 
    """
    avo = 6.02214076e23 # mol^-1
    # Reshape the grid coordinates and potential values
    x_coords = framework_grid[:, :, :, 0].flatten()
    y_coords = framework_grid[:, :, :, 1].flatten()
    z_coords = framework_grid[:, :, :, 2].flatten()
    potentials = external_pot_grid.flatten()
    num_grid_points = len(x_coords)
    grid_point_volume = volume_unit_cell_A3 / num_grid_points # in Å³

    if type == "ExtPot":
        greater_than=False
    elif type == "density":
        greater_than=True
    else:
        raise Exception(f"Error: type {type} not recognized")
    
    if threshold is not None:
        # Apply threshold to filter data
        if greater_than:
            mask = potentials >= threshold
        else:
            mask = potentials <= threshold
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        z_coords = z_coords[mask]
        potentials = potentials[mask]

    if type == "density":
        title = 'Density [mol/m^3]'
        subtitle_n = cif.split('.')[0] + ' with ' + adsorbate + ' @ ' + str(temperature) + ' K, ' + str(pressure) + ' bar'
        if segment_idx >= 0:
            subtitle_n += f' (segment {segment_idx})'
        subtitle2 = f'mean density {external_pot_grid.mean():.6g} mol/m³. molecules in unit cell: {external_pot_grid.sum() * avo * 1e-30 * grid_point_volume:.3g}'
        fname = 'density_' 
        fname += cif.split('.')[0] + '@' + str(int(np.rint(temperature))) + 'K_' + str(int(pressure*1000)) + 'mbar'
    else:
        title = 'LJ Potential [kJ/molK]'
        subtitle_n = cif.split('.')[0] + ' with ' + adsorbate + ' @ ' + str(temperature) + ' K'
        fname = 'ext_pot_' 
        if segment_idx >= 0:
            subtitle_n += f' (segment {segment_idx})'
            fname += str(segment_idx) + '_'
        fname += cif.split('.')[0] + '@' + str(int(np.rint(temperature))) + 'K'
        density_info = False

    if subtitle is None:
        subtitle = subtitle_n

    # if subtitle is None:
    #     subtitle='@ ' + str(temperature) + ' K'

    # if title is None:
    #     fname = '@' + str(int(np.rint(temperature))) + 'K_' + str(int(np.rint(pressure*1000))) + 'mbar'
    #     title = 'LJ Potential [kJ/molK]'
    # else:
    #     fname = '@' + str(int(np.rint(temperature))) + 'K_' + str(int(np.rint(pressures[-1]*1000))) + 'mbar_density'
    #     subtitle = subtitle + '@' + str(int(np.rint(temperature))) + 'K_' + ', ' + str(pressures[-1]) + ' bar'

    # Create 3D scatter plot using Plotly
    scatter_plot = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=3, # size of the markers (points)
            color=potentials,  # Color by potential values
            colorscale='Cividis' if type == 'density' else 'Plasma',#'Viridis',
            colorbar=dict(title='[kJ/molK]') if type == 'ExtPot' else dict(title='mol/m^3') ,#dict(title=title),
            opacity=0.8
        ),
        text=[f'Potential: {p:.2f} kJ/mol' for p in potentials] if type == 'ExtPot' else [f'density: {p:.2f} mol/m^3' for p in potentials],  # Add potential values to hover text
        hoverinfo='x+y+z+text'  # Show x, y, z, and potential values on hover
    )

    # Calculate the aspect ratio based on the range of coordinates
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    z_range = np.max(z_coords) - np.min(z_coords)
    max_range = max(x_range, y_range, z_range)

    aspect_ratio = dict(x=x_range / max_range, y=y_range / max_range, z=z_range / max_range)

    # Set layout for 3D plot
    layout = go.Layout(
        title={
            'text': f"{title}<br><sup>{subtitle}</sup><br><sub>{subtitle2}</sub>" if density_info else f"{title}<br><sup>{subtitle}</sup>",
            'x': 0.5
        },
        scene=dict(
            xaxis_title='X [Å]',
            yaxis_title='Y [Å]',
            zaxis_title='Z [Å]',
            camera=dict(
            # eye=dict(x=1.25, y=1.25, z=1.25),  # Adjust the initial camera position
            eye=dict(x=x, y=y, z=z),  # Adjust the initial camera position
            up=dict(x=0, y=0, z=1),  # Define the up direction
            center=dict(x=0, y=0, z=0)  # Define the center of the scene
            ),
            aspectratio=aspect_ratio, #dict(x=1, y=1, z=1), ## Adjust the aspect ratio
            aspectmode='manual'
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins to reduce whitespace
        scene_camera=dict(
            projection=dict(type='orthographic')  # Use orthographic projection for better zoom control
        )
    )
    
    config = {
        'toImageButtonOptions': {
            'filename': fname,  # Set your desired filename here
            'format': 'png'  # You can also specify 'svg', 'jpeg', etc.
        }
    }

    # Create figure and display it
    fig = go.Figure(data=[scatter_plot], layout=layout)
    fig.update_layout(
        width=x_range*y_range*4,  # Set the width of the plot
        height=z_range*40  # Set the height of the plot
    )
    pio.show(fig, config=config)



## copy of plot_3d_LJpotential_density (then adapted)
def plot_external_potential_3D(framework_grid, external_pot_grid, temperature, volume_unit_cell_A3, cif:str, adsorbate:str, pressure=None, segment_idx:int = -1, type:str="ExtPot", threshold=None, title=None, subtitle=None, density_info=False, \
                            x=1.25, y=1.25, z=1.25):
    """
    Plot the external potential or densities of a given system in 3D using Plotly.
    Inputs:
    - framework_grid: 4D NumPy array containing the coordinates of the framework grid points
    - external_pot_grid: 3D NumPy array containing the external potential values [kJ/molK] or densities [mol/m^3] at each grid point
    - temperature: temperature of the system [K]
    - pressure: pressure of the system [bar]
    - volume_unit_cell_A3: volume of the unit cell in Å³
    - cif: cif file of the system [eg RSM0016.cif]
    - adsorbate: name of the adsorbate
    - segment_idx: index of the segment to plot (optional)
    - type: type of data to plot, either 'ExtPot' for external potential or 'density' for densities
    - threshold: threshold value to filter data (optional) 
    """
    avo = 6.02214076e23 # mol^-1
    # Reshape the grid coordinates and potential values
    x_coords = framework_grid[:, :, :, 0].flatten()
    y_coords = framework_grid[:, :, :, 1].flatten()
    z_coords = framework_grid[:, :, :, 2].flatten()
    potentials = external_pot_grid.flatten()
    num_grid_points = len(x_coords)
    grid_point_volume = volume_unit_cell_A3 / num_grid_points # in Å³

    if type == "ExtPot":
        greater_than=False
    elif type == "density":
        greater_than=True
    else:
        raise Exception(f"Error: type {type} not recognized")
    
    if threshold is not None:
        # Apply threshold to filter data
        if greater_than:
            mask = potentials >= threshold
        else:
            mask = potentials <= threshold
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        z_coords = z_coords[mask]
        potentials = potentials[mask]

    if type == "density":
        title = 'Density [mol/m^3]'
        subtitle_n = cif.split('.')[0] + ' with ' + adsorbate + ' @ ' + str(temperature) + ' K, ' + str(pressure) + ' bar'
        if segment_idx >= 0:
            subtitle_n += f' (segment {segment_idx})'
        subtitle2 = f'mean density {external_pot_grid.mean():.6g} mol/m³. molecules in unit cell: {external_pot_grid.sum() * avo * 1e-30 * grid_point_volume:.3g}'
        fname = 'density_' 
        fname += cif.split('.')[0] + '@' + str(int(np.rint(temperature))) + 'K_' + str(int(pressure*1000)) + 'mbar'
    else:
        title = 'LJ Potential [kJ/molK]'
        subtitle_n = cif.split('.')[0] + ' with ' + adsorbate + ' @ ' + str(temperature) + ' K'
        fname = 'ext_pot_' 
        if segment_idx >= 0:
            subtitle_n += f' (segment {segment_idx})'
            fname += str(segment_idx) + '_'
        fname += cif.split('.')[0] + '@' + str(int(np.rint(temperature))) + 'K'
        density_info = False

    if subtitle is None:
        subtitle = subtitle_n

    # if subtitle is None:
    #     subtitle='@ ' + str(temperature) + ' K'

    # if title is None:
    #     fname = '@' + str(int(np.rint(temperature))) + 'K_' + str(int(np.rint(pressure*1000))) + 'mbar'
    #     title = 'LJ Potential [kJ/molK]'
    # else:
    #     fname = '@' + str(int(np.rint(temperature))) + 'K_' + str(int(np.rint(pressures[-1]*1000))) + 'mbar_density'
    #     subtitle = subtitle + '@' + str(int(np.rint(temperature))) + 'K_' + ', ' + str(pressures[-1]) + ' bar'

    # Create 3D scatter plot using Plotly
    scatter_plot = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=3, # size of the markers (points)
            color=potentials,  # Color by potential values
            colorscale='Cividis' if type == 'density' else 'Plasma',#'Viridis',
            colorbar=dict(title='[kJ/molK]') if type == 'ExtPot' else dict(title='mol/m^3') ,#dict(title=title),
            opacity=0.8
        ),
        text=[f'Potential: {p:.2f} kJ/mol' for p in potentials] if type == 'ExtPot' else [f'density: {p:.2f} mol/m^3' for p in potentials],  # Add potential values to hover text
        hoverinfo='x+y+z+text'  # Show x, y, z, and potential values on hover
    )

    # Calculate the aspect ratio based on the range of coordinates
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    z_range = np.max(z_coords) - np.min(z_coords)
    max_range = max(x_range, y_range, z_range)

    aspect_ratio = dict(x=x_range / max_range, y=y_range / max_range, z=z_range / max_range)

    # Set layout for 3D plot
    layout = go.Layout(
        title={
            'text': f"{title}<br><sup>{subtitle}</sup><br><sub>{subtitle2}</sub>" if density_info else f"{title}<br><sup>{subtitle}</sup>",
            'x': 0.5
        },
        scene=dict(
            xaxis_title='X [Å]',
            yaxis_title='Y [Å]',
            zaxis_title='Z [Å]',
            camera=dict(
            # eye=dict(x=1.25, y=1.25, z=1.25),  # Adjust the initial camera position
            eye=dict(x=x, y=y, z=z),  # Adjust the initial camera position
            up=dict(x=0, y=0, z=1),  # Define the up direction
            center=dict(x=0, y=0, z=0)  # Define the center of the scene
            ),
            aspectratio=aspect_ratio, #dict(x=1, y=1, z=1), ## Adjust the aspect ratio
            aspectmode='manual'
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins to reduce whitespace
        scene_camera=dict(
            projection=dict(type='orthographic')  # Use orthographic projection for better zoom control
        )
    )
    
    config = {
        'toImageButtonOptions': {
            'filename': fname,  # Set your desired filename here
            'format': 'png'  # You can also specify 'svg', 'jpeg', etc.
        }
    }

    # Create figure and display it
    fig = go.Figure(data=[scatter_plot], layout=layout)
    fig.update_layout(
        width=x_range*y_range*4,  # Set the width of the plot
        height=z_range*40  # Set the height of the plot
    )
    pio.show(fig, config=config)