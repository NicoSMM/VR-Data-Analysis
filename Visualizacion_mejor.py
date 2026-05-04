import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
CSV_PATH = "Sample data/task_01_trial_01_001_gaze.csv"
MAX_RAYS_3D = 400
RAY_LENGTH = 0.35

# =========================
# LOAD
# =========================
def load_gaze_csv(path):
    df = pd.read_csv(path)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    text_cols = [
        "timestamp_utc_iso",
        "participant_id",
        "session_id",
        "task_id",
        "trial_id",
        "condition",
        "hit_object_name",
        "hit_aoi",
        "hit_aoi_type",
    ]

    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp_rel_s").reset_index(drop=True)

    df["dt"] = df["timestamp_rel_s"].diff()
    df["dt"] = df["dt"].fillna(df["dt"].median())

    df["pupil_mean"] = df[["left_pupil_diameter", "right_pupil_diameter"]].mean(axis=1)
    df["eye_openness_mean"] = df[["left_eye_openness", "right_eye_openness"]].mean(axis=1)

    # Si no hay AOI, usar objeto como AOI aproximada
    df["aoi_or_object"] = df["hit_aoi"]
    df["aoi_or_object"] = df["aoi_or_object"].fillna(df["hit_object_name"])
    df["aoi_or_object"] = df["aoi_or_object"].fillna("None")

    return df


df = load_gaze_csv(CSV_PATH)




def plot_3d_gaze_interactive(df, max_rays=400, ray_length=0.35):
    fig = go.Figure()

    # =========================
    # Trayectoria de cabeza
    # =========================
    if all(c in df.columns for c in ["head_x", "head_y", "head_z"]):
        head_df = df.dropna(subset=["head_x", "head_y", "head_z"])

        fig.add_trace(go.Scatter3d(
            x=head_df["head_x"],
            y=head_df["head_y"],
            z=head_df["head_z"],
            mode="lines",
            name="Head trajectory",
            line=dict(width=4),
            hovertemplate=
                "Head<br>X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>"
        ))

    # =========================
    # Puntos de impacto del gaze
    # =========================
    needed_hit = ["hit_x", "hit_y", "hit_z", "hit_valid", "aoi_or_object"]
    if all(c in df.columns for c in needed_hit):
        hp = df[df["hit_valid"] == 1].dropna(subset=["hit_x", "hit_y", "hit_z"]).copy()

        if len(hp) > 0:
            fig.add_trace(go.Scatter3d(
                x=hp["hit_x"],
                y=hp["hit_y"],
                z=hp["hit_z"],
                mode="markers",
                name="Gaze hit points",
                marker=dict(
                    size=3,
                    opacity=0.55,
                    color=hp["timestamp_rel_s"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Time (s)")
                ),
                text=hp["aoi_or_object"],
                hovertemplate=
                    "Object/AOI: %{text}<br>" +
                    "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>"
            ))

    # =========================
    # Rayos de mirada
    # =========================
    needed_ray = [
        "combined_origin_x", "combined_origin_y", "combined_origin_z",
        "combined_dir_x", "combined_dir_y", "combined_dir_z",
        "combined_valid"
    ]

    if all(c in df.columns for c in needed_ray):
        gd = df[df["combined_valid"] == 1].dropna(subset=needed_ray).copy()

        if len(gd) > 0:
            step = max(1, len(gd) // max_rays)
            gd = gd.iloc[::step].copy()

            ox = gd["combined_origin_x"].to_numpy()
            oy = gd["combined_origin_y"].to_numpy()
            oz = gd["combined_origin_z"].to_numpy()

            dx = gd["combined_dir_x"].to_numpy()
            dy = gd["combined_dir_y"].to_numpy()
            dz = gd["combined_dir_z"].to_numpy()

            # Normalizar dirección
            norm = np.sqrt(dx**2 + dy**2 + dz**2)
            dx = dx / norm
            dy = dy / norm
            dz = dz / norm

            ex = ox + ray_length * dx
            ey = oy + ray_length * dy
            ez = oz + ray_length * dz

            # Plotly no tiene quiver directo simple, se grafican segmentos
            x_lines, y_lines, z_lines = [], [], []

            for i in range(len(gd)):
                x_lines += [ox[i], ex[i], None]
                y_lines += [oy[i], ey[i], None]
                z_lines += [oz[i], ez[i], None]

            fig.add_trace(go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                name="Gaze rays",
                line=dict(width=2),
                opacity=0.35,
                hoverinfo="skip"
            ))

    fig.update_layout(
        title="Visualización 3D interactiva: cabeza, rayos de mirada y puntos de impacto",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=1000,
        height=800,
        legend=dict(
            x=0.02,
            y=0.98
        )
    )

    fig.show()


plot_3d_gaze_interactive(df, MAX_RAYS_3D, RAY_LENGTH)



def plot_3d_hits_by_object(df, max_objects=15):
    hp = df[df["hit_valid"] == 1].dropna(subset=["hit_x", "hit_y", "hit_z"]).copy()

    if len(hp) == 0:
        print("No hay puntos hit válidos.")
        return

    top_objects = hp["aoi_or_object"].value_counts().head(max_objects).index
    hp = hp[hp["aoi_or_object"].isin(top_objects)]

    fig = go.Figure()

    for obj in top_objects:
        sub = hp[hp["aoi_or_object"] == obj]

        fig.add_trace(go.Scatter3d(
            x=sub["hit_x"],
            y=sub["hit_y"],
            z=sub["hit_z"],
            mode="markers",
            name=str(obj),
            marker=dict(size=3, opacity=0.65),
            hovertemplate=
                f"Object/AOI: {obj}<br>" +
                "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>"
        ))

    fig.update_layout(
        title="Puntos de impacto 3D agrupados por objeto/AOI",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=1000,
        height=800
    )

    fig.show()


plot_3d_hits_by_object(df)




def compute_dwell_time(df):
    hits = df[df["hit_valid"] == 1].copy()

    if len(hits) == 0:
        return pd.DataFrame()

    dwell = (
        hits.groupby("aoi_or_object")
        .agg(
            dwell_time_s=("dt", "sum"),
            n_samples=("dt", "count"),
            pupil_mean=("pupil_mean", "mean"),
            pupil_std=("pupil_mean", "std"),
            eye_openness_mean=("eye_openness_mean", "mean")
        )
        .reset_index()
        .sort_values("dwell_time_s", ascending=False)
    )

    total_time = dwell["dwell_time_s"].sum()
    dwell["dwell_time_pct"] = 100 * dwell["dwell_time_s"] / total_time

    return dwell


dwell = compute_dwell_time(df)
print(dwell.head(20))


import matplotlib.pyplot as plt

def plot_object_sequence(df, top_n=15):
    hits = df[df["hit_valid"] == 1].copy()

    if len(hits) == 0:
        print("No hay hits válidos.")
        return

    top_objects = hits["aoi_or_object"].value_counts().head(top_n).index
    seq_df = hits[hits["aoi_or_object"].isin(top_objects)].copy()

    object_to_y = {obj: i for i, obj in enumerate(top_objects)}

    plt.figure(figsize=(14, 6))
    plt.scatter(
        seq_df["timestamp_rel_s"],
        seq_df["aoi_or_object"].map(object_to_y),
        s=8,
        alpha=0.7
    )

    plt.yticks(list(object_to_y.values()), list(object_to_y.keys()))
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Objeto/AOI")
    plt.title("Secuencia temporal de objetos/AOIs mirados")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()


plot_object_sequence(df)



def compute_object_transitions(df):
    temp = df.copy()
    temp["obj"] = temp["aoi_or_object"].fillna("None")

    # Eliminar None si quieres solo objetos reales
    temp = temp[temp["obj"] != "None"].copy()

    if len(temp) == 0:
        return pd.DataFrame(), 0

    temp["prev_obj"] = temp["obj"].shift(1)
    transitions = temp[temp["obj"] != temp["prev_obj"]].copy()

    transition_table = (
        transitions.groupby(["prev_obj", "obj"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    n_transitions = len(transitions)

    return transition_table, n_transitions


transition_table, n_transitions = compute_object_transitions(df)

print(f"Número total de transiciones: {n_transitions}")
print(transition_table.head(20))



def compute_gaze_angular_velocity(df):
    valid = df[df["combined_valid"] == 1].copy()

    needed = ["combined_dir_x", "combined_dir_y", "combined_dir_z", "timestamp_rel_s"]
    valid = valid.dropna(subset=needed)

    dirs = valid[["combined_dir_x", "combined_dir_y", "combined_dir_z"]].to_numpy()
    t = valid["timestamp_rel_s"].to_numpy()

    norms = np.linalg.norm(dirs, axis=1)
    dirs = dirs / norms[:, None]

    dot = np.sum(dirs[1:] * dirs[:-1], axis=1)
    dot = np.clip(dot, -1, 1)

    angle = np.degrees(np.arccos(dot))
    dt = np.diff(t)

    vel = angle / dt
    vel = np.insert(vel, 0, np.nan)

    valid["gaze_ang_vel_deg_s"] = vel

    return valid


gaze_vel_df = compute_gaze_angular_velocity(df)

plt.figure(figsize=(14, 4))
plt.plot(gaze_vel_df["timestamp_rel_s"], gaze_vel_df["gaze_ang_vel_deg_s"], linewidth=0.8)
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad angular (deg/s)")
plt.title("Velocidad angular de la mirada")
plt.grid(True)
plt.tight_layout()
plt.show()



def detect_fixations_ivt(df, vel_threshold=30, min_duration=0.1):
    gaze = compute_gaze_angular_velocity(df)

    gaze["is_fixation"] = gaze["gaze_ang_vel_deg_s"] < vel_threshold
    gaze["fix_change"] = gaze["is_fixation"] != gaze["is_fixation"].shift()
    gaze["fix_segment"] = gaze["fix_change"].cumsum()

    fix = (
        gaze[gaze["is_fixation"]]
        .groupby("fix_segment")
        .agg(
            start_time=("timestamp_rel_s", "first"),
            end_time=("timestamp_rel_s", "last"),
            duration_s=("timestamp_rel_s", lambda x: x.max() - x.min()),
            n_samples=("timestamp_rel_s", "count"),
            mean_pupil=("pupil_mean", "mean"),
            mean_hit_x=("hit_x", "mean"),
            mean_hit_y=("hit_y", "mean"),
            mean_hit_z=("hit_z", "mean"),
            object_or_aoi=("aoi_or_object", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
        )
        .reset_index(drop=True)
    )

    fix = fix[fix["duration_s"] >= min_duration].copy()

    return fix


fixations = detect_fixations_ivt(df, vel_threshold=30, min_duration=0.1)

print(fixations.head())
print(f"N fijaciones: {len(fixations)}")
print(f"Duración media fijación: {fixations['duration_s'].mean():.3f} s")



def plot_fixations_3d(fixations):
    fix = fixations.dropna(subset=["mean_hit_x", "mean_hit_y", "mean_hit_z"]).copy()

    if len(fix) == 0:
        print("No hay fijaciones con coordenadas 3D.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=fix["mean_hit_x"],
        y=fix["mean_hit_y"],
        z=fix["mean_hit_z"],
        mode="markers",
        marker=dict(
            size=fix["duration_s"] * 30,
            color=fix["duration_s"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Duración (s)"),
            opacity=0.75
        ),
        text=fix["object_or_aoi"],
        hovertemplate=
            "Objeto/AOI: %{text}<br>" +
            "Duración=%{marker.color:.3f} s<br>" +
            "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
        name="Fixations"
    ))

    fig.update_layout(
        title="Fijaciones estimadas en espacio 3D",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=1000,
        height=800
    )

    fig.show()


plot_fixations_3d(fixations)


def summarize_trial(df, fixations=None):
    duration = df["timestamp_rel_s"].max() - df["timestamp_rel_s"].min()

    summary = {
        "participant_id": df["participant_id"].iloc[0],
        "session_id": df["session_id"].iloc[0],
        "task_id": df["task_id"].iloc[0],
        "trial_id": df["trial_id"].iloc[0],
        "condition": df["condition"].iloc[0],
        "duration_s": duration,
        "fs_est_hz": 1 / df["timestamp_rel_s"].diff().median(),
        "combined_valid_pct": 100 * df["combined_valid"].mean(),
        "left_valid_pct": 100 * df["left_valid"].mean(),
        "right_valid_pct": 100 * df["right_valid"].mean(),
        "hit_valid_pct": 100 * df["hit_valid"].mean(),
        "pupil_mean": df["pupil_mean"].mean(),
        "pupil_std": df["pupil_mean"].std(),
        "eye_openness_mean": df["eye_openness_mean"].mean(),
        "eye_openness_std": df["eye_openness_mean"].std(),
    }

    if all(c in df.columns for c in ["head_x", "head_y", "head_z"]):
        head_disp = np.sqrt(
            df["head_x"].diff()**2 +
            df["head_y"].diff()**2 +
            df["head_z"].diff()**2
        )

        summary["head_movement_total"] = head_disp.sum()
        summary["head_x_std"] = df["head_x"].std()
        summary["head_y_std"] = df["head_y"].std()
        summary["head_z_std"] = df["head_z"].std()

    if fixations is not None and len(fixations) > 0:
        summary["n_fixations"] = len(fixations)
        summary["fixation_duration_mean"] = fixations["duration_s"].mean()
        summary["fixation_duration_std"] = fixations["duration_s"].std()
        summary["total_fixation_time"] = fixations["duration_s"].sum()
        summary["fixation_rate_per_s"] = len(fixations) / duration

    transition_table, n_transitions = compute_object_transitions(df)
    summary["n_object_transitions"] = n_transitions
    summary["transition_rate_per_s"] = n_transitions / duration

    return pd.DataFrame([summary])


summary_trial = summarize_trial(df, fixations)
print(summary_trial.T)