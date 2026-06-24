## bene 2026-06-19: copied from /Users/bene/Documents/Projects/epse/master-thesis-local-code/final_server_files/thesis/paper-3d_dft_feos_saft/plot_ExtPot_densities.py

#%% plot_ext-pot&densities.py
# author: Benedikt Buhk
# date: 13.12.2024
# description: plot the external potential and/or densities of a given system (MOF, adsorbate,  + pressure, temperature for densities)

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from .geometry import get_angle_between_vectors
from .tols_colors import get_cmap
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
def plot_external_potential_3D(grid_xyz, external_potential_K=None, temperature_K = None, lattice=None,
                                   upper_bound=None, lower_bound=None, 
                                   overwrite_title=None, subtitle="", 
                                   #x_cam=1.25, y_cam=1.25, z_cam=1.25,
                                    #x_cam=-1.75, y_cam=-1.75, z_cam=-0.75,
                                    #x_cam=-1.44, y_cam=-1.45, z_cam=-0.15,
                                    x_cam=-1.44, y_cam=-1.45, z_cam=0.015,
                                    marker_size=1,
                                   cubic_frame_min=-10,
                                   cubic_frame=40, # if not None, set the x,y,z range to [cubic_frame_min, cubic_frame] and aspectmode to 'cube' (equal-length axes)
                                   plot_cube=0, # if >0, plot cube with given edge length (in Å) at origin for distance reference
                                   width=800, height=600,
                                   save_path=None,
                                   ):
    """
    Plot the external potential of a given system in 3D using Plotly.
    Inputs:
    - grid_xyz: 4D NumPy array containing the coordinates of the grid points (shape: (nx, ny, nz, 3))
    - external_potential_K: 3D NumPy array containing the external potential values [K] at each grid point (shape: (nx, ny, nz))
    - temperature_K: temperature in Kelvin, used to convert potential values to reduced units       
    - lattice: 2D NumPy array containing the lattice vectors (shape: (3, 3)), optional (will give infos like angles and volume)
    - upper_bound: upper bound for potential values to include in the plot (optional, will filter out points with potential > upper_bound)
    - lower_bound: lower bound for potential values to include in the plot (optional, will filter out points with potential < lower_bound)
    - overwrite_title: if provided, use this string as the title of the plot instead of the default "External Potential V^ext [K]"
    - subtitle: additional subtitle text to include below the title (default: empty string)
    - x_cam, y_cam, z_cam: camera position for the 3D plot (default: x=-1.44, y=-1.45, z=0.015 for a good view of the MOF-adsorbate system)
    - marker_size: size of the markers (points) in the scatter plot (default: 1)
    - cubic_frame_min, cubic_frame: if cubic_frame is not None, set the x,y,z range to [cubic_frame_min, cubic_frame] and aspectmode to 'cube' (equal-length axes)
    - plot_cube: int, if >0, plot a cube with the given edge length (in Å) at origin for distance reference, optional
    - width, height: dimensions of the plot in pixels (default: 800x600)
    - save_path: if provided, save the plot to this path (supports .png, .svg, .pdf, .html)
    Outputs:
    - Displays the 3D scatter plot of the external potential values at the grid points,
        colored by the potential values and with hover text showing the potential at each point.
    - If save_path is provided, also saves the plot to the specified file.
    """
    
    # Reshape the grid coordinates and potential values
    x_coords = grid_xyz[:, :, :, 0].flatten()
    y_coords = grid_xyz[:, :, :, 1].flatten()
    z_coords = grid_xyz[:, :, :, 2].flatten()
    num_grid_points = len(x_coords)
    grid_info_text = "<br># grid points: " + str(num_grid_points) + f" (a1: {int(grid_xyz.shape[0])}, a2: {int(grid_xyz.shape[1])}, a3: {int(grid_xyz.shape[2])})"

    energy_unit = "[K]"
    if temperature_K is not None:
        energy_unit = f"(ul @{temperature_K} K)"
    title=f"External Potential V<sup>ext</sup> {energy_unit}"
    show_colorbar = True
    if external_potential_K is None:
        external_potential_K = np.ones(grid_xyz.shape[:-1])*np.nan
        title="Grid points (no external potential)"
        subtitle += grid_info_text
        show_colorbar = False
        if upper_bound is not None or lower_bound is not None:
            print("WARNING: upper_bound and lower_bound are ignored since external_potential is None.")
            upper_bound = None
            lower_bound = None
    if overwrite_title is not None:
        title = overwrite_title

    if lattice is not None:
        volume_unit_cell_A3 = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
        grid_point_volume = volume_unit_cell_A3 / num_grid_points # in Å³
        #lattice_info_text = f"lattice: |a1|={np.linalg.norm(lattice[0]):.2f}, |a2|={np.linalg.norm(lattice[1]):.2f}, |a3|={np.linalg.norm(lattice[2]):.2f}Å," + f' V={volume_unit_cell_A3:.2f} Å³, per gp:{grid_point_volume:.6f} Å³<br>'
        #lattice_info_text = f"lattice: alpha={get_angle_between_vectors(lattice[1], lattice[2])}°, beta={get_angle_between_vectors(lattice[1], lattice[2])}°, gamma={get_angle_between_vectors(lattice[0], lattice[2])}°. f' V={volume_unit_cell_A3:.2f} Å³, per gp:{grid_point_volume:.6f} Å³ ({num_grid_points} gps)<br>"
        lattice_info_text = f"<br>lattice: α={get_angle_between_vectors(lattice[1], lattice[2]):.2f}°, β={get_angle_between_vectors(lattice[0], lattice[2]):.2f}°, γ={get_angle_between_vectors(lattice[1], lattice[0]):.2f}°. V={volume_unit_cell_A3:.2f} Å³, per gp:{grid_point_volume:.6f} Å³"#({num_grid_points} gps)<br>"
        subtitle += lattice_info_text

    potentials = external_potential_K.flatten()
    if temperature_K is not None:
        potentials = potentials / temperature_K

    # filter out points outside the specified bounds if provided
    mask = np.ones_like(potentials, dtype=bool)  # Initialize mask to include all points
    if upper_bound is not None:
        mask_up = potentials <= upper_bound
        n_filtered_up = np.sum(~mask_up)
        if n_filtered_up > 0:
            print(f"WARNING: Filtering out {n_filtered_up} points above upper_bound of {upper_bound} K")
        mask &= mask_up  # Combine with existing mask
    if lower_bound is not None:
        mask_low = potentials >= lower_bound
        n_filtered_low = np.sum(~mask_low)
        if n_filtered_low > 0:
            print(f"WARNING: Filtering out {n_filtered_low} points below lower_bound of {lower_bound} K")
        mask &= mask_low  # Combine with existing mask

    x_coords = x_coords[mask]
    y_coords = y_coords[mask]
    z_coords = z_coords[mask]
    potentials = potentials[mask]

    # Create 3D scatter plot using Plotly
    scatter_plot = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=marker_size, # size of the markers (points)
            color=potentials,  # Color by potential values
            #colorscale='Viridis', #'Cividis' 'Plasma',#'Viridis',
            #colorscale=sunset_scale,
            colorscale=[
                [i / 255, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
                for i, (r, g, b, a) in enumerate(get_cmap('sunset')(np.linspace(0, 1, 256)))
            ],
            # colorscale=[  # paul tols rainbow
            #     [i / 33, c] for i, c in enumerate([
            #         "#E8ECFB", "#DDD8EF", "#D1C1E1", "#C3A8D1", "#B58FC2", "#A778B4",
            #         "#9B62A7", "#8C4E99", "#6F4C9B", "#6059A9", "#5568B8", "#4E79C5",
            #         "#4D8AC6", "#4E96BC", "#549EB3", "#59A5A9", "#60AB9E", "#69B190",
            #         "#77B77D", "#8CBC68", "#A6BE54", "#BEBC48", "#D1B541", "#DDAA3C",
            #         "#E49C39", "#E78C35", "#E97A31", "#E7652F", "#E34F2B", "#DD3D2D",
            #         "#D22D2D", "#C11E31", "#AB1636", "#911539"
            #     ])
            # ], #'Cividis' 'Plasma',#'Viridis',
            cmax=upper_bound if upper_bound is not None else potentials.min(),
            cmin=lower_bound if lower_bound is not None else potentials.max(),
            showscale=show_colorbar,
            colorbar=dict(
                        title=f"V<sup>ext</sup> {energy_unit}", 
                        x=0.0, # move colorbar to the right
                        xanchor="left",
                        bgcolor="rgba(0,0,0,0)",      # transparent background
                        #bgcolor="rgba(255,0,0,0.4)", # semi-transparent white background
                        #borderwidth=0,
                        outlinewidth=0,              # drop the border box
                        borderwidth=0,
                        len=0.8,           # optional: shorter bar
                        thickness=15,      # optional: thinner
                        ),
             opacity=0.8,
            ),
        text=[f"V<sup>ext</sup>: {p:.2f} {energy_unit}" for p in potentials], # Add potential values to hover text
        name=f'{num_grid_points} grid points' if not show_colorbar else f'V<sup>ext</sup> ({num_grid_points} gps)',
        hoverinfo='x+y+z+text'  # Show x, y, z, and potential values on hover
    )
    
    #plot a cube with edge length of 10 Å at origin for distance reference if desired
    cube_traces = []
    if plot_cube:
        cube_size = plot_cube  # edge length of the cube in Å
        cube_x = [0, cube_size, cube_size, 0, 0, cube_size, cube_size, 0]
        cube_y = [0, 0, cube_size, cube_size, 0, 0, cube_size, cube_size]
        cube_z = [0, 0, 0, 0, cube_size, cube_size, cube_size, cube_size]
        # define the edges of the cube (pairs of vertices that form edges)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4), # top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
        ]
        edge_x = []
        edge_y = []
        edge_z = []
        for start, end in edges:
            edge_x += [cube_x[start], cube_x[end], None] # None to create a break in the line after each edge
            edge_y += [cube_y[start], cube_y[end], None]
            edge_z += [cube_z[start], cube_z[end], None]
        cube_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=2),
            name=f'{cube_size} Å reference',
            showlegend=True,
            hoverinfo='skip'
        )
        cube_traces = [cube_trace]

    # add lattice vectors as lines if provided
    lattice_traces = []
    if lattice is not None:
        origin = np.array([0, 0, 0])
        colors = ["#BB5566", "#228833", "#004488"] # colors for a1, a2, a3 (red, green, blue)

        for i, vec in enumerate(lattice):
            end_point = origin + vec
            line = go.Scatter3d(
                x=[origin[0], end_point[0]],
                y=[origin[1], end_point[1]],
                z=[origin[2], end_point[2]],
                mode='lines',
                line=dict(color=colors[i], width=5),
                name=f'a{i+1} (||={np.linalg.norm(vec):.2f} Å)',
                legendgroup=f'a{i+1}',      # <- group id
                showlegend=True,
                hoverinfo='skip'
            )
            lattice_traces.append(line)

            # add cones at the end of each lattice vector to indicate direction
            cone = go.Cone(
                x=[end_point[0]], y=[end_point[1]], z=[end_point[2]],
                u=[vec[0]], v=[vec[1]], w=[vec[2]],
                sizemode="absolute",
                sizeref=2.5,  # Adjust the size of the cones as needed
                anchor="tip",
                colorscale=[[0, colors[i]], [1, colors[i]]],
                showscale=False,
                legendgroup=f'a{i+1}',      # <- same group
                showlegend=False,           # <- no separate entry
                hoverinfo='skip'
            )
            lattice_traces.append(cone)


        a1, a2, a3 = lattice[0], lattice[1], lattice[2]
        # the 7 corners beyond the origin
        # edges: pairs of corners connected by one lattice vector
        box_edges = [
            # top three (from origin) are already drawn as a1,a2,a3
            (origin, a1), (origin, a2), (origin, a3),
            (a1, a1 + a2), (a1, a1 + a3),
            (a2, a2 + a1), (a2, a2 + a3),
            (a3, a3 + a1), (a3, a3 + a2),
            (a1 + a2, a1 + a2 + a3),
            (a1 + a3, a1 + a3 + a2),
            (a2 + a3, a2 + a3 + a1),
        ]
        box_x, box_y, box_z = [], [], []
        for start, end in box_edges:
            box_x += [start[0], end[0], None]
            box_y += [start[1], end[1], None]
            box_z += [start[2], end[2], None]
        box_trace = go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode='lines',
            line=dict(color='gray', width=2),
            name=f'unit cell ({volume_unit_cell_A3:.2f} Å³)',
            showlegend=True,
            hoverinfo='skip'
        )
        lattice_traces.append(box_trace)
        
    # Calculate the aspect ratio based on the range of coordinates
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    z_range = np.max(z_coords) - np.min(z_coords)
    max_range = max(x_range, y_range, z_range)

    if max_range == 0:           # single point (or all points coincident)
        aspect_ratio = dict(x=1, y=1, z=1)
    else:
        aspect_ratio = dict(
            x=x_range / max_range,
            y=y_range / max_range,
            z=z_range / max_range,
        )

    # Set layout for 3D plot
    layout = go.Layout(
        title={
            'text': f"{title}<br><sup>{subtitle}</sup>",
            'x': 0.5
        },
        scene=dict(
            xaxis_title='X [Å]',
            yaxis_title='Y [Å]',
            zaxis_title='Z [Å]',
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),   # <- fixed horizontal placement (of scene)
            #domain=dict(x=[0.30, 0.82], y=[0.0, 0.88]),  # pull right edge in from 1.0, top in from 1.0
            camera=dict(
            # eye=dict(x=1.25, y=1.25, z=1.25),  # Adjust the initial camera position
            eye=dict(x=x_cam, y=y_cam, z=z_cam),  # Adjust the initial camera position
            up=dict(x=0, y=0, z=1),  # Define the up direction
            center=dict(x=0, y=0, z=0)  # Define the center of the scene
            ),
            aspectratio=aspect_ratio, #dict(x=1, y=1, z=1), ## Adjust the aspect ratio
            aspectmode='manual',
            #xaxis=dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False), # change axis background to transparent
            #yaxis=dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False), # change axis background to transparent
            #zaxis=dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False), # change axis background to transparent
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins to reduce whitespace
        # scene_camera=dict(
        #     projection=dict(type='orthographic')  # Use orthographic projection for better zoom control
        # )
    )

    fname = 'external_potential_3D'
    
    config = {
        'toImageButtonOptions': {
            'filename': fname,  # Set your desired filename here
            'format': 'png'  # You can also specify 'svg', 'jpeg', etc.
        }
    }

    # Create figure and display it
    fig = go.Figure(data=[scatter_plot] + lattice_traces + cube_traces, layout=layout)
    # fig.update_layout(
    #     width=x_range*y_range*4,  # Set the width of the plot
    #     height=z_range*40  # Set the height of the plot
    # )
    if cubic_frame is not None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[cubic_frame_min, cubic_frame]),
                yaxis=dict(range=[cubic_frame_min, cubic_frame]),
                zaxis=dict(range=[cubic_frame_min, cubic_frame]),
                aspectmode='cube',  # equal-length axes -> correct relative sizing
            ),
        )
    fig.update_layout(
        legend=dict(
            x=0.85, xanchor="left",   # inside the right portion of the plot
            y=0.95, yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        # legend=dict(
        #     bgcolor="rgba(0,0,0,0)",
        #     bordercolor="rgba(0,0,0,0)",
        #     borderwidth=0,
        # ),
        #paper_bgcolor="rgba(0,0,0,0)",   # whole figure canvas transparent
        #plot_bgcolor="rgba(0,0,0,0)",   # <- this one
    )
    
    fig.update_layout(width=width, height=height)  # fixed size to keep caption readable
    pio.show(fig, config=config)

    if save_path is not None:
        ext = save_path.lower().rsplit(".", 1)[-1]
        if ext in {"svg", "pdf", "png"}:
            try:
                fig.write_image(save_path, width=800, height=600)
            except Exception as e:
                html_path = save_path.rsplit(".", 1)[0] + ".html"
                fig.write_html(html_path)
                print(f"Static export failed ({e}); wrote {html_path} instead. "
                    f"For SVG/PDF: pip install 'plotly>=6.1.1' and run "
                    f"kaleido_get_chrome.")
        elif ext in {"html", "htm"}:
            fig.write_html(save_path)
        else:
            raise ValueError(f"unsupported save_path extension: {ext}")
        print(f"Wrote {save_path}")


## copy of plot_3d_LJpotential_density (then adapted)
def plot_external_potential_3D_with_histogram(grid_xyz, external_potential_K=None, temperature_K = None, lattice=None,
                                   upper_bound=None, lower_bound=None, 
                                   overwrite_title=None, subtitle="", 
                                   #x_cam=1.25, y_cam=1.25, z_cam=1.25,
                                    #x_cam=-1.75, y_cam=-1.75, z_cam=-0.75,
                                    #x_cam=-1.44, y_cam=-1.45, z_cam=-0.15,
                                    x_cam=-1.44, y_cam=-1.45, z_cam=0.015,
                                    marker_size=1,
                                   cubic_frame_min=-10,
                                   cubic_frame=40, # if not None, set the x,y,z range to [cubic_frame_min, cubic_frame] and aspectmode to 'cube' (equal-length axes)
                                   plot_cube=0, # if >0, plot cube with given edge length (in Å) at origin for distance reference
                                   width=800, height=600,
                                   save_path=None,
                                   show_histogram=False, nbins = 50,
                                   y_lo_hist=0.02, y_hi_hist=0.94,
                                   x_center_scene=-0.0, y_center_scene=0.0,
                                   mark_min_on_colorbar=True,  
                                   show_legend=True,                             
                                   ):
    """
    Plot the external potential of a given system in 3D using Plotly.
    Inputs:
    - grid_xyz: 4D NumPy array containing the coordinates of the grid points (shape: (nx, ny, nz, 3))
    - external_potential_K: 3D NumPy array containing the external potential values [K] at each grid point (shape: (nx, ny, nz))
    - temperature_K: temperature in Kelvin, used to convert potential values to reduced units       
    - lattice: 2D NumPy array containing the lattice vectors (shape: (3, 3)), optional (will give infos like angles and volume)
    - upper_bound: upper bound for potential values to include in the plot (optional, will filter out points with potential > upper_bound)
    - lower_bound: lower bound for potential values to include in the plot (optional, will filter out points with potential < lower_bound)
    - overwrite_title: if provided, use this string as the title of the plot instead of the default "External Potential V^ext [K]"
    - subtitle: additional subtitle text to include below the title (default: empty string)
    - x_cam, y_cam, z_cam: camera position for the 3D plot (default: x=-1.44, y=-1.45, z=0.015 for a good view of the MOF-adsorbate system)
    - marker_size: size of the markers (points) in the scatter plot (default: 1)
    - cubic_frame_min, cubic_frame: if cubic_frame is not None, set the x,y,z range to [cubic_frame_min, cubic_frame] and aspectmode to 'cube' (equal-length axes)
    - plot_cube: int, if >0, plot a cube with the given edge length (in Å) at origin for distance reference, optional
    - width, height: dimensions of the plot in pixels (default: 800x600)
    - save_path: if provided, save the plot to this path (supports .png, .svg, .pdf, .html)
    Outputs:
    - Displays the 3D scatter plot of the external potential values at the grid points,
        colored by the potential values and with hover text showing the potential at each point.
    - If save_path is provided, also saves the plot to the specified file.
    """
    
    # Reshape the grid coordinates and potential values
    x_coords = grid_xyz[:, :, :, 0].flatten()
    y_coords = grid_xyz[:, :, :, 1].flatten()
    z_coords = grid_xyz[:, :, :, 2].flatten()
    num_grid_points = len(x_coords)
    grid_info_text = "<br># grid points: " + str(num_grid_points) + f" (a1: {int(grid_xyz.shape[0])}, a2: {int(grid_xyz.shape[1])}, a3: {int(grid_xyz.shape[2])})"

    energy_unit = "[K]"
    if temperature_K is not None:
        energy_unit = f"(ul @{temperature_K} K)"
    title=f"External Potential V<sup>ext</sup> {energy_unit}"
    exp_pot_provided = True
    if external_potential_K is None:
        external_potential_K = np.ones(grid_xyz.shape[:-1])*np.nan
        title="Grid points (no external potential)"
        subtitle += grid_info_text
        exp_pot_provided = False
        if upper_bound is not None or lower_bound is not None:
            print("WARNING: upper_bound and lower_bound are ignored since external_potential is None.")
            upper_bound = None
            lower_bound = None
        if mark_min_on_colorbar:
            print("WARNING: mark_min_on_colorbar=True but no external_potential is provided. Ignoring mark_min_on_colorbar.")
            mark_min_on_colorbar = False
    if overwrite_title is not None:
        title = overwrite_title

    if lattice is not None:
        volume_unit_cell_A3 = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
        grid_point_volume = volume_unit_cell_A3 / num_grid_points # in Å³
        #lattice_info_text = f"lattice: |a1|={np.linalg.norm(lattice[0]):.2f}, |a2|={np.linalg.norm(lattice[1]):.2f}, |a3|={np.linalg.norm(lattice[2]):.2f}Å," + f' V={volume_unit_cell_A3:.2f} Å³, per gp:{grid_point_volume:.6f} Å³<br>'
        #lattice_info_text = f"lattice: alpha={get_angle_between_vectors(lattice[1], lattice[2])}°, beta={get_angle_between_vectors(lattice[1], lattice[2])}°, gamma={get_angle_between_vectors(lattice[0], lattice[2])}°. f' V={volume_unit_cell_A3:.2f} Å³, per gp:{grid_point_volume:.6f} Å³ ({num_grid_points} gps)<br>"
        lattice_info_text = f"<br>lattice: α={get_angle_between_vectors(lattice[1], lattice[2]):.2f}°, β={get_angle_between_vectors(lattice[0], lattice[2]):.2f}°, γ={get_angle_between_vectors(lattice[1], lattice[0]):.2f}°. V={volume_unit_cell_A3:.2f} Å³, per gp:{grid_point_volume:.6f} Å³"#({num_grid_points} gps)<br>"
        subtitle += lattice_info_text

    potentials = external_potential_K.flatten()
    if temperature_K is not None:
        potentials = potentials / temperature_K

    # filter out points outside the specified bounds if provided
    mask = np.ones_like(potentials, dtype=bool)  # Initialize mask to include all points
    if upper_bound is not None:
        mask_up = potentials <= upper_bound
        n_filtered_up = np.sum(~mask_up)
        if n_filtered_up > 0:
            print(f"WARNING: Filtering out {n_filtered_up} points above upper_bound of {upper_bound} K")
        mask &= mask_up  # Combine with existing mask
    if lower_bound is not None:
        mask_low = potentials >= lower_bound
        n_filtered_low = np.sum(~mask_low)
        if n_filtered_low > 0:
            print(f"WARNING: Filtering out {n_filtered_low} points below lower_bound of {lower_bound} K")
        mask &= mask_low  # Combine with existing mask

    x_coords = x_coords[mask]
    y_coords = y_coords[mask]
    z_coords = z_coords[mask]
    potentials = potentials[mask]
    has_data = len(potentials) > 0
    if not has_data:
        print("WARNING: No grid points remain after applying bounds "
            f"(lower_bound={lower_bound}, upper_bound={upper_bound}). Nothing to plot.")
        subtitle += "<br>WARNING: No grid points remain after applying bounds."
        #return


    if show_histogram and exp_pot_provided:
        # decide the energy range the histogram should span (match colorbar)
        e_lo = lower_bound if lower_bound is not None else np.nanmin(potentials)
        e_hi = upper_bound if upper_bound is not None else np.nanmax(potentials)

        # build the histogram on the host side
        counts, edges = np.histogram(potentials[~np.isnan(potentials)],
                                    bins=nbins, range=(e_lo, e_hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        #print(f"centers: {centers} (shape: {centers.shape}). edges: {edges} (shape: {edges.shape}))")
        n_in_hist = int(counts.sum())
        dE_hist = edges[1] - edges[0]
        hist_trace = [go.Bar(
            x=counts,
            y=centers,
            orientation='h',
            marker=dict(
                #color='black',                       # color bars by energy too
                color=centers,                       # color bars by energy too
                colorscale=[
                    [i / 255, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
                    for i, (r, g, b, a) in enumerate(get_cmap('sunset')(np.linspace(1,0, 256)))
                ],
                cmin=e_lo, cmax=e_hi,
            ),
            xaxis='x2', yaxis='y2',
            showlegend=False,
            hovertemplate="V<sup>ext</sup>: %{y:.2f} {energy_unit}<br># gp: %{x}<extra></extra>",
        )]
    else: 
        if show_histogram and not exp_pot_provided:
            print("WARNING: show_histogram=True but no external_potential is provided. Please provide external_potential to show histogram of potential values.")
            show_histogram = False
        hist_trace = []

    # Create 3D scatter plot using Plotly
    scatter_plot = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
            showlegend=True,
        marker=dict(
            size=marker_size, # size of the markers (points)
            color=potentials,  # Color by potential values
            #colorscale='Viridis', #'Cividis' 'Plasma',#'Viridis',
            #colorscale=sunset_scale,
            colorscale=[
                [i / 255, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
                for i, (r, g, b, a) in enumerate(get_cmap('sunset')(np.linspace(1, 0, 256)))
            ],
            # colorscale=[  # paul tols rainbow
            #     [i / 33, c] for i, c in enumerate([
            #         "#E8ECFB", "#DDD8EF", "#D1C1E1", "#C3A8D1", "#B58FC2", "#A778B4",
            #         "#9B62A7", "#8C4E99", "#6F4C9B", "#6059A9", "#5568B8", "#4E79C5",
            #         "#4D8AC6", "#4E96BC", "#549EB3", "#59A5A9", "#60AB9E", "#69B190",
            #         "#77B77D", "#8CBC68", "#A6BE54", "#BEBC48", "#D1B541", "#DDAA3C",
            #         "#E49C39", "#E78C35", "#E97A31", "#E7652F", "#E34F2B", "#DD3D2D",
            #         "#D22D2D", "#C11E31", "#AB1636", "#911539"
            #     ])
            # ], #'Cividis' 'Plasma',#'Viridis',
            cmax=upper_bound if upper_bound is not None else potentials.max(),
            cmin=lower_bound if lower_bound is not None else potentials.min(),
            showscale=exp_pot_provided,
            colorbar=dict(
                        title=f"V<sup>ext</sup> {energy_unit}", 
                        x=0.0, # move colorbar to the right
                        xanchor="left",
                        bgcolor="rgba(0,0,0,0)",      # transparent background
                        #bgcolor="rgba(255,0,0,0.4)", # semi-transparent white background
                        #borderwidth=0,
                        outlinewidth=0,              # drop the border box
                        borderwidth=0,
                        len=1.0,           # optional: shorter bar (whatch out with histogram alignment if enabled)
                        thickness=15,      # optional: thinner
                        tickmode="auto",          # keep default ticks
                        # add the min as an extra annotated tick:
                        tickvals=None,            # let auto handle the rest
                        ),
             opacity=0.8,
            ),
        text=[f"V<sup>ext</sup>: {p:.2f} {energy_unit} [gp: {i}]" for i, p in enumerate(potentials)], # Add potential values to hover text
        name=f'{num_grid_points} grid points' if not exp_pot_provided else f'V<sup>ext</sup> ({num_grid_points} gps)',
        hoverinfo='x+y+z+text'  # Show x, y, z, and potential values on hover
    )
    
    #plot a cube with edge length of 10 Å at origin for distance reference if desired
    cube_traces = []
    if plot_cube:
        cube_size = plot_cube  # edge length of the cube in Å
        cube_x = [0, cube_size, cube_size, 0, 0, cube_size, cube_size, 0]
        cube_y = [0, 0, cube_size, cube_size, 0, 0, cube_size, cube_size]
        cube_z = [0, 0, 0, 0, cube_size, cube_size, cube_size, cube_size]
        # define the edges of the cube (pairs of vertices that form edges)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4), # top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
        ]
        edge_x = []
        edge_y = []
        edge_z = []
        for start, end in edges:
            edge_x += [cube_x[start], cube_x[end], None] # None to create a break in the line after each edge
            edge_y += [cube_y[start], cube_y[end], None]
            edge_z += [cube_z[start], cube_z[end], None]
        cube_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=2),
            name=f'{cube_size} Å reference',
            showlegend=True,
            hoverinfo='skip'
        )
        cube_traces = [cube_trace]

    # add lattice vectors as lines if provided
    lattice_traces = []
    if lattice is not None:
        origin = np.array([0, 0, 0])
        colors = ["#BB5566", "#228833", "#004488"] # colors for a1, a2, a3 (red, green, blue)

        for i, vec in enumerate(lattice):
            end_point = origin + vec
            line = go.Scatter3d(
                x=[origin[0], end_point[0]],
                y=[origin[1], end_point[1]],
                z=[origin[2], end_point[2]],
                mode='lines',
                line=dict(color=colors[i], width=5),
                name=f'a{i+1} (||={np.linalg.norm(vec):.2f} Å)',
                legendgroup=f'a{i+1}',      # <- group id
                showlegend=True,
                hoverinfo='skip'
                #hoverinfo=f'x+y+z'  # Show end point coordinates on hover
            )
            lattice_traces.append(line)

            # add cones at the end of each lattice vector to indicate direction
            cone = go.Cone(
                x=[end_point[0]], y=[end_point[1]], z=[end_point[2]],
                u=[vec[0]], v=[vec[1]], w=[vec[2]],
                sizemode="absolute",
                sizeref=2.5,  # Adjust the size of the cones as needed
                anchor="tip",
                colorscale=[[0, colors[i]], [1, colors[i]]],
                #color=colors[i],
                showscale=False,
                legendgroup=f'a{i+1}',      # <- same group
                showlegend=False,           # <- no separate entry
                hoverinfo=f'x+y+z',  # Show end point coordinates on hover
                #hoverinfo='skip',
                hoverlabel=dict(
                    bgcolor=colors[i],            # tooltip background in the vector's color
                    font=dict(color="white"),     # readable text on the colored background
                    bordercolor=colors[i],
                ),
            )
            lattice_traces.append(cone)


        a1, a2, a3 = lattice[0], lattice[1], lattice[2]
        # the 7 corners beyond the origin
        # edges: pairs of corners connected by one lattice vector
        box_edges = [
            # top three (from origin) are already drawn as a1,a2,a3
            (origin, a1), (origin, a2), (origin, a3),
            (a1, a1 + a2), (a1, a1 + a3),
            (a2, a2 + a1), (a2, a2 + a3),
            (a3, a3 + a1), (a3, a3 + a2),
            (a1 + a2, a1 + a2 + a3),
            (a1 + a3, a1 + a3 + a2),
            (a2 + a3, a2 + a3 + a1),
        ]
        box_x, box_y, box_z = [], [], []
        for start, end in box_edges:
            box_x += [start[0], end[0], None]
            box_y += [start[1], end[1], None]
            box_z += [start[2], end[2], None]
        box_trace = go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode='lines',
            line=dict(color='gray', width=2),
            name=f'unit cell ({volume_unit_cell_A3:.2f} Å³)',
            showlegend=True,
            hoverinfo='skip'
        )
        lattice_traces.append(box_trace)
        
    
    # Calculate the aspect ratio based on the range of coordinates
    if has_data:
        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)
        z_range = np.max(z_coords) - np.min(z_coords)
        max_range = max(x_range, y_range, z_range)
        if max_range == 0:
            aspect_ratio = dict(x=1, y=1, z=1)
        else:
            aspect_ratio = dict(x=x_range/max_range, y=y_range/max_range, z=z_range/max_range)
    else:
        aspect_ratio = dict(x=1, y=1, z=1)   # nothing to scale to


    # Set layout for 3D plot
    layout = go.Layout(
        title={
            'text': f"{title}<br><sup>{subtitle}</sup>",
            'x': 0.5
        },
        scene=dict(
            xaxis_title='X [Å]',
            yaxis_title='Y [Å]',
            zaxis_title='Z [Å]',
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),   # <- fixed horizontal placement (of scene)
            #domain=dict(x=[0.30, 0.82], y=[0.0, 0.88]),  # pull right edge in from 1.0, top in from 1.0
            camera=dict(
            # eye=dict(x=1.25, y=1.25, z=1.25),  # Adjust the initial camera position
            eye=dict(x=x_cam, y=y_cam, z=z_cam),  # Adjust the initial camera position
            up=dict(x=0, y=0, z=1),  # Define the up direction
            center=dict(x=x_center_scene, y=y_center_scene, z=0)  # Define the center of the scene
            ),
            aspectratio=aspect_ratio, #dict(x=1, y=1, z=1), ## Adjust the aspect ratio
            aspectmode='manual',
            #xaxis=dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False), # change axis background to transparent
            #yaxis=dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False), # change axis background to transparent
            #zaxis=dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False), # change axis background to transparent
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins to reduce whitespace
        # scene_camera=dict(
        #     projection=dict(type='orthographic')  # Use orthographic projection for better zoom control
        # )
        xaxis2=dict( # for histogram
            domain=[0.1, 0.25],          # left strip width
            anchor='y2',
            #title=f'# gp (sum: {n_in_hist})<br>{nbins} a {dE_hist:.2f} {energy_unit}', # add total count to axis title
            title=dict(
                text=f'# gp (sum: {n_in_hist})<br>{nbins} a {dE_hist:.2f} {energy_unit}',
                font=dict(size=12),       # <- title text size
            ),
            autorange='reversed',        # counts grow leftward, toward colorbar
            side='top',
        ) if show_histogram else None,
        yaxis2=dict( # for histogram
            domain=[y_lo_hist, y_hi_hist],           # vertical span — match colorbar `len`/position
            anchor='x2',
            range=[e_lo, e_hi],          # <- aligns with colorbar energy axis
            side='right',
            showticklabels=False,        # colorbar already shows the energy ticks
        ) if show_histogram else None,
    )

    fname = 'external_potential_3D'
    
    config = {
        'toImageButtonOptions': {
            'filename': fname,  # Set your desired filename here
            'format': 'png'  # You can also specify 'svg', 'jpeg', etc.
        }
    }

    # Create figure and display it
    fig = go.Figure(data=[scatter_plot] + hist_trace + lattice_traces + cube_traces, layout=layout)
    # fig.update_layout(
    #     width=x_range*y_range*4,  # Set the width of the plot
    #     height=z_range*40  # Set the height of the plot
    # )
    if cubic_frame is not None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[cubic_frame_min, cubic_frame]),
                yaxis=dict(range=[cubic_frame_min, cubic_frame]),
                zaxis=dict(range=[cubic_frame_min, cubic_frame]),
                aspectmode='cube',  # equal-length axes -> correct relative sizing
            ),
        )
    fig.update_layout(
        legend=dict(
            x=0.85, xanchor="left",   # inside the right portion of the plot
            y=0.95, yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        showlegend=show_legend,
        # legend=dict(
        #     bgcolor="rgba(0,0,0,0)",
        #     bordercolor="rgba(0,0,0,0)",
        #     borderwidth=0,
        # ),
        #paper_bgcolor="rgba(0,0,0,0)",   # whole figure canvas transparent
        #plot_bgcolor="rgba(0,0,0,0)",   # <- this one
    )
    if has_data and mark_min_on_colorbar: # works for height=600
        if height != 600:
            print("WARNING: plot_min_on_colorbar=True is tuned for height=600. If you change the height, the annotation position may be off. Adjust y_lo_hist/y_hi_hist or disable plot_min_on_colorbar to fix.")
        # add a horizontal line at the minimum potential value on the colorbar
        # add annotation for the minimum potential value on the colorbar
        v_min = float(np.nanmin(potentials))
        v_min_idx = int(np.nanargmin(potentials))
        cmin_val = lower_bound if lower_bound is not None else potentials.min()
        cmax_val = upper_bound if upper_bound is not None else potentials.max()
        # fraction of the bar height where v_min sits
        frac = (v_min - cmin_val) / (cmax_val - cmin_val) if cmax_val != cmin_val else 0.0
        # colorbar with len=1.0, default y=0.5 yanchor middle -> spans paper y [0,1]
        #bar_y = frac           # adjust if you change len/y of the colorbar
        offset_bottom= y_lo_hist #+ 0.01
        bar_len = (y_hi_hist - offset_bottom)          # whatever you set on the colorbar
        bar_center = 0.5       # colorbar `y`
        #bar_y = (bar_center - bar_len/2) + frac 
        bar_y = frac * bar_len + offset_bottom
        #bar_y = (bar_y + offset_bottom)
        fig.add_annotation(
            x=0.07, y=bar_y,           # x just right of the bar (bar at x=0.0, thickness~15px)
            xref="paper", yref="paper",
            text=f"min: {v_min:.2f} {energy_unit} [gp: {v_min_idx}]",
            showarrow=True,
            arrowhead=3, arrowsize=1, arrowwidth=1.5,
            ax=40, ay=0,               # arrow points left toward the bar
            font=dict(size=9, color="black"),
            bgcolor="rgba(255,255,255,0.6)",
        )
        fig.add_shape(
            type="line",
            x0=0.0, x1=0.1,          # bar x-position to bar right edge (tune to thickness)
            y0=bar_y, y1=bar_y,
            xref="paper", yref="paper",
            line=dict(color="black", width=2),
        )
    
    fig.update_layout(width=width, height=height)  # fixed size to keep caption readable
    pio.show(fig, config=config)

    if save_path is not None:
        ext = save_path.lower().rsplit(".", 1)[-1]
        if ext in {"svg", "pdf", "png"}:
            try:
                fig.write_image(save_path, width=width, height=height)
            except Exception as e:
                html_path = save_path.rsplit(".", 1)[0] + ".html"
                fig.write_html(html_path)
                print(f"Static export failed ({e}); wrote {html_path} instead. "
                    f"For SVG/PDF: pip install 'plotly>=6.1.1' and run "
                    f"kaleido_get_chrome.")
        elif ext in {"html", "htm"}:
            fig.write_html(save_path)
        else:
            raise ValueError(f"unsupported save_path extension: {ext}")
        print(f"Wrote {save_path}")

