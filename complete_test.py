import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq

# =========================
# CONFIGURACIÓN
# =========================
#file_path = r"Tomas\Live Data Recording 3_0.txt"
file_path = r"Test_15_05_2026\Sujeto1piolotaje150420256_1_0.txt"
#event_file_path = r"Tomas\Live Data Recording 3_0_event2.txt"
event_file_path = r"Test_15_05_2026\Sujeto1piolotaje150420256_1_0_event.txt"
signals = [
    "MWMOBILE2_ECG",
    "MWMOBILE2_Z0",
    "MWMOBILE2_Resp",
    "MWMOBILE2_EDA"
]

# Parámetros ajustables
ECG_BANDPASS = (5, 20)          # Hz, resalta complejos QRS del ECG
DZDT_BANDPASS = (0.8, 20)       # Hz, deja componente cardíaca dominante de dZ/dt
Z0_LOWPASS = 0.7                # Hz, para ver componente lenta de Z0
RESP_BAND = (0.05, 0.7)         # Hz, banda típica respiratoria en Z0
DZDT_MIN_DISTANCE = 0.4         # s, distancia mínima entre peaks cardíacos
EDA_SMOOTH_WINDOW_SEC = 0.5     # s, suavizado EDA
EDA_MIN_PROMINENCE = None       # si quieres fijarla manualmente, usa un valor numérico

SHOW_EVENT_LABELS = True        # mostrar texto sobre cada evento
EVENT_ALPHA = 0.7               # transparencia de línea de evento
EVENT_COLOR = "purple"          # color de línea de evento
EVENT_LINESTYLE = "--"          # estilo de línea

# =========================
# FUNCIONES AUXILIARES
# =========================
def butter_filter(signal, fs, cutoff, btype='low', order=4):
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, signal)

def moving_average(x, window_samples):
    window_samples = max(1, int(window_samples))
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(x, kernel, mode="same")

def estimate_fs(time_s):
    dt = np.median(np.diff(time_s))
    return 1.0 / dt

def safe_std(x):
    return np.std(x, ddof=1) if len(x) > 1 else np.nan

def load_events(event_file_path):
    events_df = pd.read_csv(event_file_path, sep="\t")
    events_df.columns = events_df.columns.str.strip()

    required_cols = ["Event Type", "Name", "Time"]
    for col in required_cols:
        if col not in events_df.columns:
            raise ValueError(f"No se encontró la columna de eventos: {col}")

    events_df["Time"] = pd.to_numeric(events_df["Time"], errors="coerce")
    events_df = events_df.dropna(subset=["Time"]).reset_index(drop=True)
    return events_df

def add_events_to_axis(ax, events_df, show_labels=True, color="purple",
                       linestyle="--", alpha=0.7, fontsize=8):
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    for _, row in events_df.iterrows():
        t = row["Time"]
        label = str(row["Name"])

        ax.axvline(t, color=color, linestyle=linestyle, alpha=alpha)

        if show_labels:
            ax.text(
                t,
                ymax - 0.02 * yrange,
                label,
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=fontsize,
                color=color,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
            )

def add_events_to_axes(axs, events_df, show_labels=True, color="purple",
                       linestyle="--", alpha=0.7, fontsize=8):
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    # primero dejamos que el eje ya tenga su escala definida
    for ax in axs:
        add_events_to_axis(
            ax,
            events_df,
            show_labels=show_labels,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            fontsize=fontsize
        )

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(file_path, sep="\t", skiprows=1)
df.columns = df.columns.str.strip()

print("Columnas encontradas:")
print(df.columns.tolist())

required_cols = ["Time (s)"] + signals
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"No se encontró la columna: {col}")

events_df = load_events(event_file_path)

print("\nEventos encontrados:")
print(events_df)

time_s = df["Time (s)"].to_numpy()
fs = estimate_fs(time_s)
print(f"\nFrecuencia de muestreo estimada: {fs:.2f} Hz")

ecg = df["MWMOBILE2_ECG"].to_numpy()
z0 = df["MWMOBILE2_Resp"].to_numpy()
dzdt = df["MWMOBILE2_Z0"].to_numpy()
eda = df["MWMOBILE2_EDA"].to_numpy()

# =========================
# 1) GRÁFICOS BASE
# =========================
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

for ax, signal in zip(axs, signals):
    ax.plot(df["Time (s)"], df[signal], linewidth=1)
    ax.set_title(signal)
    ax.set_ylabel("Amplitud")
    ax.grid(True)

axs[-1].set_xlabel("Tiempo (s)")
add_events_to_axes(axs, events_df, show_labels=SHOW_EVENT_LABELS,
                   color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, alpha=EVENT_ALPHA)
plt.tight_layout()
plt.show()

# =========================
# 2) ECG: DETECCIÓN DE R-PEAKS + HRV
# =========================
ecg_filt = butter_filter(ecg, fs, ECG_BANDPASS, btype='bandpass', order=3)

distance_samples_ecg = int(0.4 * fs)
prom_ecg = 0.5 * np.std(ecg_filt)

r_peaks, _ = find_peaks(
    ecg_filt,
    distance=distance_samples_ecg,
    prominence=prom_ecg
)

r_times = time_s[r_peaks]
rr = np.diff(r_times)

if len(rr) >= 2:
    hr_inst = 60.0 / rr
    mean_hr = np.mean(hr_inst)
    mean_rr = np.mean(rr)

    sdnn = safe_std(rr) * 1000

    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr**2)) * 1000 if len(diff_rr) > 0 else np.nan
    pnn50 = 100 * np.mean(np.abs(diff_rr) > 0.05) if len(diff_rr) > 0 else np.nan
else:
    hr_inst = np.array([])
    mean_hr = mean_rr = sdnn = rmssd = pnn50 = np.nan

print("\n===== HRV desde ECG =====")
print(f"Número de R-peaks detectados: {len(r_peaks)}")
print(f"Frecuencia cardiaca media: {mean_hr:.2f} bpm")
print(f"RR medio: {mean_rr*1000:.2f} ms")
print(f"SDNN: {sdnn:.2f} ms")
print(f"RMSSD: {rmssd:.2f} ms")
print(f"pNN50: {pnn50:.2f} %")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time_s, ecg, label="ECG crudo", alpha=0.4)
ax.plot(time_s, ecg_filt, label="ECG filtrado", linewidth=1.2)
ax.plot(r_times, ecg_filt[r_peaks], "ro", label="R-peaks")
ax.set_title("ECG con detección de R-peaks")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Amplitud")
ax.grid(True)
ax.legend()
add_events_to_axes(ax, events_df, show_labels=SHOW_EVENT_LABELS,
                   color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, alpha=EVENT_ALPHA)
plt.tight_layout()
plt.show()

if len(rr) > 0:
    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    axs[0].plot(r_times[1:], rr * 1000, marker='o')
    axs[0].set_title("Intervalos RR")
    axs[0].set_xlabel("Tiempo (s)")
    axs[0].set_ylabel("RR (ms)")
    axs[0].grid(True)

    axs[1].plot(r_times[1:], hr_inst, marker='o')
    axs[1].set_title("Frecuencia cardíaca instantánea")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("HR (bpm)")
    axs[1].grid(True)

    add_events_to_axes(axs, events_df, show_labels=SHOW_EVENT_LABELS,
                       color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, alpha=EVENT_ALPHA)
    plt.tight_layout()
    plt.show()

# =========================
# 3) Z0: FFT + FILTRO PASA-BAJOS
# =========================
z0_centered = z0 - np.mean(z0)
freqs_z0 = rfftfreq(len(z0_centered), d=1/fs)
fft_z0 = np.abs(rfft(z0_centered))

z0_low = butter_filter(z0, fs, Z0_LOWPASS, btype='low', order=4)

fig, axs = plt.subplots(3, 1, figsize=(14, 10))

axs[0].plot(time_s, z0, label="Z0 cruda")
axs[0].set_title("Z0 cruda")
axs[0].set_xlabel("Tiempo (s)")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)

axs[1].plot(freqs_z0, fft_z0)
axs[1].set_title("FFT de Z0")
axs[1].set_xlabel("Frecuencia (Hz)")
axs[1].set_ylabel("Magnitud")
axs[1].set_xlim(0, min(5, fs/2))
axs[1].grid(True)

axs[2].plot(time_s, z0, alpha=0.4, label="Z0 cruda")
axs[2].plot(time_s, z0_low, linewidth=2, label=f"Z0 filtrada LPF {Z0_LOWPASS} Hz")
axs[2].set_title("Z0 filtrada pasa-bajos")
axs[2].set_xlabel("Tiempo (s)")
axs[2].set_ylabel("Amplitud")
axs[2].grid(True)
axs[2].legend()

# solo tiene sentido poner eventos en gráficos temporales, no en FFT
add_events_to_axes([axs[0], axs[2]], events_df, show_labels=SHOW_EVENT_LABELS,
                   color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, alpha=EVENT_ALPHA)

plt.tight_layout()
plt.show()

# =========================
# 3B) Z0: FRECUENCIA RESPIRATORIA (ROBUSTA)
# =========================
z0_lp = butter_filter(z0, fs, Z0_LOWPASS, btype='low', order=4)
z0_lp = z0_lp - np.mean(z0_lp)
dz0 = np.gradient(z0_lp, time_s)

sign_dz0 = np.sign(dz0)
sign_change = np.diff(sign_dz0)

peaks_der = np.where(sign_change < 0)[0] + 1
mins_der = np.where(sign_change > 0)[0] + 1

min_breath_distance = int(1.2 * fs)
prominence_z0 = 0.2 * np.std(z0_lp)

def refine_peaks(indices, signal, fs, prominence):
    refined = []
    for idx in indices:
        left = max(0, idx - int(1.5 * fs))
        right = min(len(signal), idx + int(1.5 * fs))

        local_min = np.min(signal[left:right])
        if (signal[idx] - local_min) >= prominence:
            refined.append(idx)
    return np.array(refined, dtype=int)

def enforce_distance(indices, signal, min_dist):
    if len(indices) == 0:
        return indices

    selected = [indices[0]]
    for idx in indices[1:]:
        if idx - selected[-1] >= min_dist:
            selected.append(idx)
        else:
            if signal[idx] > signal[selected[-1]]:
                selected[-1] = idx
    return np.array(selected, dtype=int)

resp_peaks = refine_peaks(peaks_der, z0_lp, fs, prominence_z0)
resp_peaks = enforce_distance(resp_peaks, z0_lp, min_breath_distance)

resp_mins = refine_peaks(mins_der, -z0_lp, fs, prominence_z0)
resp_mins = enforce_distance(resp_mins, -z0_lp, min_breath_distance)

resp_peak_times = time_s[resp_peaks]

if len(resp_peak_times) >= 2:
    breath_intervals = np.diff(resp_peak_times)
    rr_breath = 60.0 / breath_intervals
    mean_rr_breath = np.mean(rr_breath)
else:
    rr_breath = np.array([])
    mean_rr_breath = np.nan

print("\n===== Z0: frecuencia respiratoria (derivada + LPF) =====")
print(f"Peaks detectados: {len(resp_peaks)}")
print(f"Frecuencia media: {mean_rr_breath:.2f} resp/min")

fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axs[0].plot(time_s, z0_lp, label="Z0 low-pass")
axs[0].plot(time_s[resp_peaks], z0_lp[resp_peaks], "ro", label="Peaks")
axs[0].plot(time_s[resp_mins], z0_lp[resp_mins], "go", label="Mínimos")
axs[0].set_title("Z0 respiratoria (LPF + derivada)")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(time_s, dz0, label="Derivada")
axs[1].axhline(0, linestyle="--")
axs[1].set_title("Derivada de Z0")
axs[1].set_xlabel("Tiempo (s)")
axs[1].set_ylabel("dZ0/dt")
axs[1].grid(True)

add_events_to_axes(axs, events_df, show_labels=SHOW_EVENT_LABELS,
                   color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, alpha=EVENT_ALPHA)

plt.tight_layout()
plt.show()

# =========================
# 4) dZ/dt: FILTRADO + DETECCIÓN DE PEAKS + MÉTRICAS PROXY
# =========================
dzdt_filt = butter_filter(dzdt, fs, DZDT_BANDPASS, btype='bandpass', order=3)

smooth_window = max(1, int(0.03 * fs))
dzdt_smooth = moving_average(dzdt_filt, smooth_window)

dzdt_proc = dzdt_smooth.copy()

distance_samples_dzdt = int(DZDT_MIN_DISTANCE * fs)
height_dzdt = np.mean(dzdt_proc) + 0.8 * np.std(dzdt_proc)
prom_dzdt = 0.5 * np.std(dzdt_proc)
min_width_dzdt = max(1, int(0.015 * fs))

dzdt_peaks, dzdt_props = find_peaks(
    dzdt_proc,
    distance=distance_samples_dzdt,
    prominence=prom_dzdt,
    height=height_dzdt,
    width=min_width_dzdt
)

dzdt_peak_times = time_s[dzdt_peaks]
dzdt_peak_vals = dzdt_proc[dzdt_peaks]

min_peaks, _ = find_peaks(-dzdt_proc, distance=max(1, int(0.15 * fs)))

lvet_list = []
sv_proxy_list = []
contractility_proxy_list = []
effort_proxy_list = []
valid_peak_times = []
valid_peak_indices = []

for p in dzdt_peaks:
    prev_mins = min_peaks[min_peaks < p]
    next_mins = min_peaks[min_peaks > p]

    if len(prev_mins) == 0 or len(next_mins) == 0:
        continue

    left = prev_mins[-1]
    right = next_mins[0]

    lvet = time_s[right] - time_s[left]
    peak_amp = abs(dzdt_proc[p])

    z0_local = np.mean(z0[max(0, left):min(len(z0), right)])
    z0_local = abs(z0_local) if abs(z0_local) > 1e-6 else 1e-6

    contractility_proxy = peak_amp
    sv_proxy = (peak_amp * lvet) / (z0_local ** 2)

    lvet_list.append(lvet)
    sv_proxy_list.append(sv_proxy)
    contractility_proxy_list.append(contractility_proxy)
    valid_peak_times.append(time_s[p])
    valid_peak_indices.append(p)

if len(r_times) >= 2 and len(valid_peak_times) > 0:
    hr_time = r_times[1:]
    hr_values = 60.0 / np.diff(r_times)

    for t_peak, svp in zip(valid_peak_times, sv_proxy_list):
        idx = np.argmin(np.abs(hr_time - t_peak))
        local_hr = hr_values[idx]
        effort_proxy_list.append(local_hr * svp)
else:
    effort_proxy_list = [np.nan] * len(sv_proxy_list)

print("\n===== dZ/dt: detección y métricas proxy =====")
print(f"Height mínimo usado: {height_dzdt:.4f}")
print(f"Prominence mínima usada: {prom_dzdt:.4f}")
print(f"Width mínimo usado: {min_width_dzdt} muestras")
print(f"Peaks detectados en dZ/dt: {len(dzdt_peaks)}")
print(f"Peaks válidos para métricas: {len(valid_peak_times)}")

if len(lvet_list) > 0:
    print(f"LVET proxy medio: {np.mean(lvet_list)*1000:.2f} ms")
    print(f"Contractility proxy media (|dZ/dt pico|): {np.mean(contractility_proxy_list):.4f}")
    print(f"SV proxy medio (relativo): {np.mean(sv_proxy_list):.6f}")
    if np.sum(~np.isnan(effort_proxy_list)) > 0:
        print(f"Effort proxy medio (HR × SV proxy): {np.nanmean(effort_proxy_list):.6f}")
else:
    print("No se pudieron calcular proxies de LVET/SV.")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time_s, dzdt, alpha=0.2, label="dZ/dt cruda")
ax.plot(time_s, dzdt_filt, alpha=0.5, label="dZ/dt filtrada")
ax.plot(time_s, dzdt_proc, linewidth=1.5, label="dZ/dt suavizada")
ax.plot(dzdt_peak_times, dzdt_peak_vals, "ro", label="Peaks detectados")
ax.axhline(height_dzdt, color="green", linestyle="--", label="Threshold height")
ax.set_title("dZ/dt filtrada con detección robusta de peaks")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Amplitud")
ax.grid(True)
ax.legend()
add_events_to_axes(ax, events_df, show_labels=SHOW_EVENT_LABELS,
                   color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, alpha=EVENT_ALPHA)
plt.tight_layout()
plt.show()

if len(sv_proxy_list) > 0:
    beat_axis = np.arange(1, len(sv_proxy_list) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axs[0].plot(beat_axis, np.array(lvet_list) * 1000, marker='o')
    axs[0].set_title("LVET proxy latido a latido")
    axs[0].set_ylabel("ms")
    axs[0].grid(True)

    axs[1].plot(beat_axis, contractility_proxy_list, marker='o')
    axs[1].set_title("Contractility proxy latido a latido")
    axs[1].set_ylabel("u.a.")
    axs[1].grid(True)

    axs[2].plot(beat_axis, effort_proxy_list, marker='o')
    axs[2].set_title("Effort proxy relativo latido a latido")
    axs[2].set_ylabel("u.a.")
    axs[2].set_xlabel("Latido")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# =========================
# 5) EDA: MÁXIMOS Y MÍNIMOS BASADOS EN DERIVADA
# =========================
eda_smooth = moving_average(eda, EDA_SMOOTH_WINDOW_SEC * fs)
deda = np.gradient(eda_smooth, time_s)

sign_deda = np.sign(deda)
sign_change = np.diff(sign_deda)

max_idx = np.where(sign_change < 0)[0] + 1
min_idx = np.where(sign_change > 0)[0] + 1

if EDA_MIN_PROMINENCE is None:
    EDA_MIN_PROMINENCE = 0.1 * np.std(eda_smooth)

valid_max = []
for idx in max_idx:
    left = max(0, idx - int(1.0 * fs))
    right = min(len(eda_smooth), idx + int(1.0 * fs))
    local_min = np.min(eda_smooth[left:right])
    if (eda_smooth[idx] - local_min) >= EDA_MIN_PROMINENCE:
        valid_max.append(idx)

valid_min = []
for idx in min_idx:
    left = max(0, idx - int(1.0 * fs))
    right = min(len(eda_smooth), idx + int(1.0 * fs))
    local_max = np.max(eda_smooth[left:right])
    if (local_max - eda_smooth[idx]) >= EDA_MIN_PROMINENCE:
        valid_min.append(idx)

valid_max = np.array(valid_max, dtype=int)
valid_min = np.array(valid_min, dtype=int)

print("\n===== EDA: máximos y mínimos =====")
print(f"Máximos detectados: {len(valid_max)}")
print(f"Mínimos detectados: {len(valid_min)}")

cut_sec = 0.5

start_idx = int(cut_sec * fs)
end_idx = len(time_s) - int(cut_sec * fs)

mask_time = (time_s >= time_s[start_idx]) & (time_s <= time_s[end_idx])

valid_max_plot = valid_max[(valid_max >= start_idx) & (valid_max <= end_idx)]
valid_min_plot = valid_min[(valid_min >= start_idx) & (valid_min <= end_idx)]

fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axs[0].plot(time_s[mask_time], eda[mask_time], alpha=0.4, label="EDA cruda")
axs[0].plot(time_s[mask_time], eda_smooth[mask_time], linewidth=2, label="EDA suavizada")
axs[0].plot(time_s[valid_max_plot], eda_smooth[valid_max_plot], "ro", label="Máximos")
axs[0].plot(time_s[valid_min_plot], eda_smooth[valid_min_plot], "go", label="Mínimos")

axs[0].set_title(f"EDA (recortada ±{cut_sec}s) con máximos y mínimos")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(time_s[mask_time], deda[mask_time], label="Derivada EDA")
axs[1].axhline(0, color="k", linestyle="--")
axs[1].set_title("Derivada de EDA (recortada)")
axs[1].set_xlabel("Tiempo (s)")
axs[1].set_ylabel("dEDA/dt")
axs[1].grid(True)

add_events_to_axes(axs, events_df, show_labels=SHOW_EVENT_LABELS,
                   color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, alpha=EVENT_ALPHA)

plt.tight_layout()
plt.show()