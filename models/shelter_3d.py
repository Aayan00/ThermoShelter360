"""3D Geometric modeling and visualization engine for ThermoShelter360.

Renders ultra-smooth, lightweight 3D architectural models of cold-climate passive
shelters using Plotly Mesh3D with full camera persistence (uirevision) and orbit controls.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

# Arctic Engineering Studio Material Shaders
MATERIAL_THEME_COLORS = {
    "Adobe/Earth": {"wall": "#B3804D", "specular": 0.1, "roughness": 0.8},
    "Stone": {"wall": "#7E8C8D", "specular": 0.2, "roughness": 0.7},
    "Brick": {"wall": "#B85D43", "specular": 0.1, "roughness": 0.7},
    "Timber": {"wall": "#A06B3E", "specular": 0.3, "roughness": 0.6},
    "Concrete": {"wall": "#94A3B8", "specular": 0.2, "roughness": 0.6},
    "Mineral Wool": {"wall": "#D4AC0D", "specular": 0.1, "roughness": 0.9},
    "EPS Insulation": {"wall": "#E2E8F0", "specular": 0.4, "roughness": 0.4},
    "Sheep Wool": {"wall": "#FADBD8", "specular": 0.1, "roughness": 0.9},
    "Glass": {"wall": "#4CC9F0", "specular": 0.9, "roughness": 0.1},
}


def _box_mesh(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float
) -> Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]:
    """Generate minimal 8-vertex 12-triangle mesh for a 3D box."""
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 4, 4]
    j = [1, 2, 5, 6, 4, 7, 5, 6, 6, 7, 5, 6]
    k = [2, 3, 6, 7, 7, 3, 2, 2, 7, 3, 1, 2]
    return x, y, z, i, j, k


def create_3d_shelter_figure(
    length: float = 6.0,
    width: float = 4.0,
    height: float = 2.8,
    wall_thickness: float = 0.25,
    window_area: float = 3.5,
    door_area: float = 1.8,
    roof_type: str = "gable",
    material_name: str = "Adobe/Earth",
    sun_azimuth: float = 180.0,
    sun_elevation: float = 35.0,
    show_sun_ray: bool = True,
    show_snow: bool = False,
    show_heat_flow: bool = False,
    cutaway_mode: bool = False,
    height_px: int = 580,
) -> go.Figure:
    """Generate a highly optimized, smoothly interactable 3D Plotly Figure.

    Features:
        - Max 8 go.Mesh3d traces for fast 60fps WebGL rendering.
        - uirevision='shelter-3d' for non-resetting camera on rerun.
        - Orbit dragmode and custom lighting.
        - Optional Snow layer, Heat flow thermal gradients, and Cutaway interior.

    Args:
        length: Building length (m).
        width: Building width (m).
        height: Wall height (m).
        wall_thickness: Wall thickness (m).
        window_area: Glazing aperture area (m2).
        door_area: Entrance door area (m2).
        roof_type: "gable", "flat", or "shed".
        material_name: Wall material name.
        sun_azimuth: Solar azimuth angle in degrees (180 = South).
        sun_elevation: Sun elevation angle above horizon.
        show_sun_ray: Render 3D incident solar radiation vector.
        show_snow: Render alpine snow accumulation layer.
        show_heat_flow: Render thermal gradient heat flux coloration.
        cutaway_mode: Open front wall for interior cutaway inspection.
        height_px: Render height in pixels.

    Returns:
        Plotly graph_objects.Figure ready for st.plotly_chart.
    """
    fig = go.Figure()

    # Determine Wall Shading
    mat_key = "Adobe/Earth"
    for k in MATERIAL_THEME_COLORS:
        if k.lower() in material_name.lower():
            mat_key = k
            break
    mat_props = MATERIAL_THEME_COLORS[mat_key]
    wall_color = "#FF6B35" if show_heat_flow else mat_props["wall"]
    roof_color = "#F0F9FF" if show_snow else ("#1A2B4A" if roof_type.lower() != "flat" else wall_color)
    glazing_color = "#4CC9F0"
    door_color = "#FFB703"

    # 1. Ground Terrain Plane (Mesh3D)
    pad = max(length, width) * 0.7
    gx0, gx1 = -pad, length + pad
    gy0, gy1 = -pad, width + pad
    ground_color = "#E0F2FE" if show_snow else "#12203A"

    fig.add_trace(
        go.Mesh3d(
            x=[gx0, gx1, gx1, gx0],
            y=[gy0, gy0, gy1, gy1],
            z=[-0.04, -0.04, -0.04, -0.04],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=ground_color,
            opacity=0.6,
            name="Ground Terrain",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 2. Main Building Body / Walls (Mesh3D)
    if cutaway_mode:
        # Cutaway: Front wall (y=0) removed to show internal room volume
        bx = [0, length, length, 0, 0, length, length, 0]
        by = [0, 0, width, width, 0, 0, width, width]
        bz = [0, 0, 0, 0, height, height, height, height]
        # Back wall (2,3,7,6), Left wall (0,3,7,4), Right wall (1,2,6,5), Floor (0,1,2,3)
        bi = [2, 2, 0, 0, 1, 1, 0, 0]
        bj = [3, 7, 3, 7, 2, 6, 1, 2]
        bk = [7, 6, 7, 4, 6, 5, 2, 3]
    else:
        bx, by, bz, bi, bj, bk = _box_mesh(0, length, 0, width, 0, height)

    fig.add_trace(
        go.Mesh3d(
            x=bx,
            y=by,
            z=bz,
            i=bi,
            j=bj,
            k=bk,
            color=wall_color,
            opacity=0.92,
            name=f"Wall Envelope ({material_name})",
            flatshading=True,
            lighting=dict(
                ambient=0.7,
                diffuse=0.8,
                roughness=mat_props["roughness"],
                specular=mat_props["specular"],
            ),
            hovertemplate=f"<b>{material_name} Envelope</b><br>Dimensions: {length}m x {width}m x {height}m<br>Thickness: {wall_thickness}m<extra></extra>",
        )
    )

    # 3. Roof Assembly (Gable, Flat, or Shed)
    if roof_type.lower() == "gable":
        ridge_h = height + width * 0.35
        rx = [0, length, length, 0, 0, length]
        ry = [0, 0, width, width, width / 2.0, width / 2.0]
        rz = [height, height, height, height, ridge_h, ridge_h]
        ri = [0, 0, 3, 3, 0, 1]
        rj = [1, 5, 2, 5, 3, 2]
        rk = [5, 4, 5, 4, 4, 5]

        fig.add_trace(
            go.Mesh3d(
                x=rx,
                y=ry,
                z=rz,
                i=ri,
                j=rj,
                k=rk,
                color=roof_color,
                opacity=0.96,
                name="Insulated Gable Roof",
                flatshading=True,
                lighting=dict(ambient=0.7, diffuse=0.8),
                hovertemplate="<b>Insulated Gable Roof</b><br>Pitch: 35% Sloped Rafters<extra></extra>",
            )
        )
    elif roof_type.lower() == "shed":
        ridge_h = height + width * 0.28
        rx = [0, length, length, 0, 0, length, length, 0]
        ry = [0, 0, width, width, 0, 0, width, width]
        rz = [height, height, ridge_h, ridge_h, height + 0.08, height + 0.08, ridge_h + 0.08, ridge_h + 0.08]
        ri = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 4, 4]
        rj = [1, 2, 5, 6, 4, 7, 5, 6, 6, 7, 5, 6]
        rk = [2, 3, 6, 7, 7, 3, 2, 2, 7, 3, 1, 2]

        fig.add_trace(
            go.Mesh3d(
                x=rx,
                y=ry,
                z=rz,
                i=ri,
                j=rj,
                k=rk,
                color=roof_color,
                opacity=0.96,
                name="Solar-Tilted Shed Roof",
                flatshading=True,
                lighting=dict(ambient=0.7, diffuse=0.8),
                hovertemplate="<b>Solar-Tilted Shed Roof</b><br>Optimized South Collector Angle<extra></extra>",
            )
        )
    else:  # Flat Ladakh parapet roof
        fx, fy, fz, fi, fj, fk = _box_mesh(-0.1, length + 0.1, -0.1, width + 0.1, height, height + 0.18)
        fig.add_trace(
            go.Mesh3d(
                x=fx,
                y=fy,
                z=fz,
                i=fi,
                j=fj,
                k=fk,
                color=roof_color,
                opacity=0.96,
                name="Ladakh Mud Flat Roof",
                flatshading=True,
                lighting=dict(ambient=0.7, diffuse=0.8),
                hovertemplate="<b>Vernacular Mud-Straw Flat Roof</b><extra></extra>",
            )
        )

    # 4. South Glazed Solar Aperture (Window)
    if window_area > 0 and not cutaway_mode:
        max_win_w = length * 0.8
        max_win_h = height * 0.7
        win_w = min(max_win_w, math.sqrt(window_area * (max_win_w / max_win_h)))
        win_h = min(max_win_h, window_area / max(0.1, win_w))
        wx0 = (length - win_w) / 2.0
        wx1 = wx0 + win_w
        wz0 = 0.75
        wz1 = min(height - 0.15, wz0 + win_h)

        wx, wy, wz, wi, wj, wk = _box_mesh(wx0, wx1, -0.06, 0.06, wz0, wz1)
        fig.add_trace(
            go.Mesh3d(
                x=wx,
                y=wy,
                z=wz,
                i=wi,
                j=wj,
                k=wk,
                color=glazing_color,
                opacity=0.8,
                name=f"Solar Glazing ({window_area:.1f} m²)",
                flatshading=True,
                lighting=dict(ambient=0.9, specular=0.8, roughness=0.1),
                hovertemplate=f"<b>South Glazed Solar Aperture</b><br>Area: {window_area:.2f} m²<br>Aperture: {win_w:.1f}m x {wz1-wz0:.1f}m<extra></extra>",
            )
        )

    # 5. Entrance Door
    if door_area > 0 and not cutaway_mode:
        door_w = 0.95
        door_h = min(height - 0.15, max(1.8, door_area / door_w))
        dx0 = max(0.2, length * 0.82 - door_w)
        dx1 = dx0 + door_w
        dx, dy, dz, di, dj, dk = _box_mesh(dx0, dx1, -0.05, 0.05, 0.0, door_h)
        fig.add_trace(
            go.Mesh3d(
                x=dx,
                y=dy,
                z=dz,
                i=di,
                j=dj,
                k=dk,
                color=door_color,
                opacity=0.95,
                name="Insulated Entrance Door",
                flatshading=True,
                hovertemplate=f"<b>Insulated Entry Door</b><br>Area: {door_area:.2f} m²<extra></extra>",
            )
        )

    # 6. Solar Sun Vector & Incident Ray
    if show_sun_ray and sun_elevation > 0:
        rad_az = math.radians(sun_azimuth)
        rad_el = math.radians(sun_elevation)
        ray_len = max(length, width, height) * 2.0

        sun_x = (length / 2.0) - ray_len * math.cos(rad_el) * math.sin(rad_az - math.pi)
        sun_y = (width / 2.0) - ray_len * math.cos(rad_el) * math.cos(rad_az - math.pi)
        sun_z = ray_len * math.sin(rad_el)

        target_x, target_y, target_z = length / 2.0, 0.0, height / 2.0

        fig.add_trace(
            go.Scatter3d(
                x=[sun_x, target_x],
                y=[sun_y, target_y],
                z=[sun_z, target_z],
                mode="lines+markers",
                line=dict(color="#FFB703", width=7),
                marker=dict(size=[14, 5], color=["#FFB703", "#FF6B35"], symbol=["circle", "diamond"]),
                name=f"Solar Ray (Az: {sun_azimuth}°, El: {sun_elevation}°)",
                hovertemplate=f"<b>Incident Solar Radiation</b><br>Azimuth: {sun_azimuth}° | Elevation: {sun_elevation}°<extra></extra>",
            )
        )

    # 7. Lightweight Corner Picker Points for Smooth Hover & Center Registration
    cx = [0, length, length, 0]
    cy = [0, 0, width, width]
    cz = [height / 2.0] * 4
    fig.add_trace(
        go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="markers",
            marker=dict(size=1, opacity=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 8. Optimized Camera, Controls, and Transparent Background
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.8, y=-1.8, z=1.2),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height_px,
        uirevision="shelter-3d",  # Critical: camera position does not reset on parameter change
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#F0F9FF", size=11),
            bgcolor="rgba(26,43,74,0.6)",
        ),
    )

    return fig
