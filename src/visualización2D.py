import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "Sample data/task_01_trial_01_001_gaze.csv"
MAX_ARROWS_3D = 300   # para no saturar el gráfico 3D
ROLLING_WIN = 15      # suavizado visual opcional

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)

# Convertir strings vacíos a NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# Intentar convertir columnas numéricas
for col in df.columns:
    if col not in ["timestamp_utc_iso", "participant_id", "session_id", "task_id",
                   "trial_id", "condition", "hit_object_name", "hit_aoi", "hit_aoi_type"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Ordenar por tiempo por seguridad
df = df.sort_values("timestamp_rel_s").reset_index(drop=True)

# Delta t
df["dt"] = df["timestamp_rel_s"].diff()

# =========================
# RESUMEN BÁSICO
# =========================
print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNAS ===")
print(df.columns.tolist())

print("\n=== MISSINGS (%) ===")
print((df.isna().mean() * 100).sort_values(ascending=False).round(2))

valid_cols = [c for c in ["combined_valid", "left_valid", "right_valid", "hit_valid"] if c in df.columns]
if valid_cols:
    print("\n=== VALID FLAGS (%) ===")
    for c in valid_cols:
        print(f"{c}: {(df[c] == 1).mean() * 100:.2f}% válidos")

# =========================
# FIGURA 1: timing + validez
# =========================
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=False)

# dt
axs[0].plot(df.index, df["dt"], linewidth=0.8)
axs[0].set_title("Intervalo entre muestras (dt)")
axs[0].set_ylabel("s")
axs[0].grid(True)

# validity
for c in ["combined_valid", "left_valid", "right_valid", "hit_valid"]:
    if c in df.columns:
        axs[1].plot(df["timestamp_rel_s"], df[c], label=c, linewidth=0.8)
axs[1].set_title("Flags de validez")
axs[1].set_ylabel("0/1")
axs[1].legend()
axs[1].grid(True)

# openness
for c in ["left_eye_openness", "right_eye_openness"]:
    if c in df.columns:
        axs[2].plot(df["timestamp_rel_s"], df[c], label=c, linewidth=0.9)
axs[2].set_title("Apertura ocular")
axs[2].set_ylabel("0-1")
axs[2].legend()
axs[2].grid(True)

# pupils
for c in ["left_pupil_diameter", "right_pupil_diameter"]:
    if c in df.columns:
        s = df[c].rolling(ROLLING_WIN, min_periods=1).mean()
        axs[3].plot(df["timestamp_rel_s"], s, label=f"{c} (smooth)", linewidth=1.0)
axs[3].set_title("Diámetro pupilar")
axs[3].set_xlabel("timestamp_rel_s")
axs[3].set_ylabel("mm")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()

# =========================
# FIGURA 2: vergence + cabeza
# =========================
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

if "vergence_angle_deg" in df.columns:
    axs[0].plot(df["timestamp_rel_s"], df["vergence_angle_deg"], linewidth=0.9)
    axs[0].set_title("Vergence angle")
    axs[0].set_ylabel("deg")
    axs[0].grid(True)

for coord in ["head_x", "head_y", "head_z"]:
    if coord in df.columns:
        axs[1].plot(df["timestamp_rel_s"], df[coord], label=coord, linewidth=0.9)
axs[1].set_title("Head position")
axs[1].set_ylabel("pos")
axs[1].legend()
axs[1].grid(True)

for coord in ["combined_origin_x", "combined_origin_y", "combined_origin_z"]:
    if coord in df.columns:
        axs[2].plot(df["timestamp_rel_s"], df[coord], label=coord, linewidth=0.9)
axs[2].set_title("Combined gaze origin")
axs[2].set_ylabel("origin")
axs[2].legend()
axs[2].grid(True)

for coord in ["combined_dir_x", "combined_dir_y", "combined_dir_z"]:
    if coord in df.columns:
        axs[3].plot(df["timestamp_rel_s"], df[coord], label=coord, linewidth=0.9)
axs[3].set_title("Combined gaze direction")
axs[3].set_xlabel("timestamp_rel_s")
axs[3].set_ylabel("dir")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()

# =========================
# FIGURA 3: AOIs
# =========================
if "hit_aoi" in df.columns and "hit_valid" in df.columns:
    aoi_df = df[df["hit_valid"] == 1].copy()

    # tiempo aproximado por AOI usando dt
    aoi_df["dt"] = aoi_df["dt"].fillna(0)
    time_by_aoi = aoi_df.groupby("hit_aoi")["dt"].sum().sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 6))
    time_by_aoi.sort_values().plot(kind="barh")
    plt.title("Tiempo aproximado mirando cada AOI")
    plt.xlabel("Tiempo acumulado (s)")
    plt.ylabel("AOI")
    plt.tight_layout()
    plt.show()

    # secuencia temporal AOI
    top_aois = time_by_aoi.index.tolist()
    seq_df = aoi_df[aoi_df["hit_aoi"].isin(top_aois)].copy()
    aoi_to_y = {aoi: i for i, aoi in enumerate(top_aois)}

    plt.figure(figsize=(14, 5))
    plt.scatter(
        seq_df["timestamp_rel_s"],
        seq_df["hit_aoi"].map(aoi_to_y),
        s=8
    )
    plt.yticks(list(aoi_to_y.values()), list(aoi_to_y.keys()))
    plt.xlabel("timestamp_rel_s")
    plt.title("Secuencia temporal de AOIs")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()

# =========================
# FIGURA 4: 3D simple con matplotlib
# =========================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# head trajectory
if all(c in df.columns for c in ["head_x", "head_y", "head_z"]):
    ax.plot(df["head_x"], df["head_y"], df["head_z"], linewidth=1.0, label="Head trajectory")

# hit points
if all(c in df.columns for c in ["hit_x", "hit_y", "hit_z", "hit_valid"]):
    hp = df[df["hit_valid"] == 1]
    ax.scatter(hp["hit_x"], hp["hit_y"], hp["hit_z"], s=6, alpha=0.5, label="Hit points")

# gaze rays subsample
needed = ["combined_origin_x", "combined_origin_y", "combined_origin_z",
          "combined_dir_x", "combined_dir_y", "combined_dir_z", "combined_valid"]
if all(c in df.columns for c in needed):
    gd = df[df["combined_valid"] == 1].copy()
    if len(gd) > 0:
        step = max(1, len(gd) // MAX_ARROWS_3D)
        gd = gd.iloc[::step].copy()

        x = gd["combined_origin_x"].values
        y = gd["combined_origin_y"].values
        z = gd["combined_origin_z"].values
        u = gd["combined_dir_x"].values
        v = gd["combined_dir_y"].values
        w = gd["combined_dir_z"].values

        ax.quiver(x, y, z, u, v, w, length=0.15, normalize=True)

ax.set_title("Vista 3D inicial: cabeza, hits y gaze rays")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()