from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def rotate_points(xy, *, angle_radians: float) -> np.ndarray:
    """Rotate ``[[x, y], ...]`` points using FastF1's circuit rotation angle."""

    points = np.asarray(xy, dtype=float)
    rot_mat = np.array(
        [
            [np.cos(angle_radians), np.sin(angle_radians)],
            [-np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )
    return np.matmul(points, rot_mat)


def _rotated_xy(frame: pd.DataFrame, angle_radians: float) -> np.ndarray:
    return rotate_points(frame.loc[:, ["X", "Y"]].to_numpy(), angle_radians=angle_radians)


def _pick_driver_lap(laps, driver: str, lap_number: int | None):
    driver_laps = laps.pick_drivers(driver) if hasattr(laps, "pick_drivers") else laps[laps["Driver"] == driver]

    if lap_number is not None:
        try:
            selected = driver_laps.pick_lap(lap_number)
            if selected is not None:
                return selected
        except Exception:
            selected_rows = driver_laps[driver_laps["LapNumber"] == lap_number]
            if not selected_rows.empty:
                return selected_rows.iloc[0]

    try:
        return driver_laps.pick_fastest()
    except Exception:
        return driver_laps.sort_values("LapTime").iloc[0] if not driver_laps.empty else None


def build_track_map(
    session,
    *,
    drivers: list[str] | None = None,
    lap_number: int | None = None,
    lap_progress: float = 0.50,
    show_corners: bool = True,
    background_color: str = "rgba(0,0,0,0)",
    text_color: str | None = None,
    driver_marker_size: int = 7,
) -> go.Figure:
    """Create an accurate circuit map with optional driver markers.

    The track shape comes from FastF1 position telemetry. Corner labels and map
    rotation come from ``session.get_circuit_info()``.
    """

    if not 0 <= lap_progress <= 1:
        raise ValueError("lap_progress must be between 0 and 1.")

    fastest_lap = session.laps.pick_fastest()
    pos = fastest_lap.get_pos_data()
    circuit_info = session.get_circuit_info()
    track_angle = circuit_info.rotation / 180 * np.pi
    track = _rotated_xy(pos, track_angle)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=track[:, 0],
            y=track[:, 1],
            mode="lines",
            name="Racing line",
            line={"width": 6},
            hoverinfo="skip",
        )
    )

    if show_corners and hasattr(circuit_info, "corners"):
        for _, corner in circuit_info.corners.iterrows():
            corner_label = f"{corner['Number']}{corner['Letter']}"
            offset_angle = corner["Angle"] / 180 * np.pi
            offset_x, offset_y = rotate_points([[500, 0]], angle_radians=offset_angle)[0]
            text_x = corner["X"] + offset_x
            text_y = corner["Y"] + offset_y

            label_x, label_y = rotate_points([[text_x, text_y]], angle_radians=track_angle)[0]
            track_x, track_y = rotate_points([[corner["X"], corner["Y"]]], angle_radians=track_angle)[0]

            fig.add_trace(
                go.Scatter(
                    x=[track_x, label_x],
                    y=[track_y, label_y],
                    mode="lines",
                    line={"width": 1, "dash": "dot"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[label_x],
                    y=[label_y],
                    mode="markers+text",
                    text=[corner_label],
                    textposition="middle center",
                    marker={"size": 24, "line": {"width": 1}},
                    name=f"Turn {corner_label}",
                    showlegend=False,
                    hoverinfo="text",
                )
            )

    if drivers:
        for driver in drivers:
            lap = _pick_driver_lap(session.laps, driver, lap_number)
            if lap is None:
                continue

            try:
                driver_pos = lap.get_pos_data()
            except Exception:
                continue

            if driver_pos.empty:
                continue

            point_idx = int(round((len(driver_pos) - 1) * lap_progress))
            point_idx = max(0, min(point_idx, len(driver_pos) - 1))
            point = _rotated_xy(driver_pos.iloc[[point_idx]], track_angle)[0]

            fig.add_trace(
                go.Scatter(
                    x=[point[0]],
                    y=[point[1]],
                    mode="markers",
                    customdata=[
                        (
                            f"<b>{driver}</b><br>"
                            f"Lap: {lap.get('LapNumber', 'Unknown')}<br>"
                            f"Team: {lap.get('Team', 'Unknown')}<br>"
                            f"Compound: {lap.get('Compound', 'Unknown')}<br>"
                            f"Progress: {lap_progress:.0%}"
                        )
                    ],
                    hovertemplate="%{customdata}<extra></extra>",
                    marker={"size": driver_marker_size, "line": {"width": 1}},
                    name=driver,
                )
            )

    event_name = session.event.get("EventName", session.event.get("Name", "Selected Grand Prix"))
    title = f"{event_name} track map"
    if lap_number is not None:
        title += f" — lap {lap_number}, {lap_progress:.0%} lap progress"

    layout = {
        "title": title,
        "xaxis": {"visible": False},
        "yaxis": {"visible": False, "scaleanchor": "x", "scaleratio": 1},
        "plot_bgcolor": background_color,
        "paper_bgcolor": background_color,
        "height": 650,
        "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        "legend_title": "Drivers",
    }

    if text_color:
        layout["font"] = {"color": text_color}

    fig.update_layout(**layout)

    return fig
