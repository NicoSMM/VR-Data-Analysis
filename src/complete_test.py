import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
from pathlib import Path

# =========================
# RUTAS
# =========================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "results" / "analisis_mwmobile"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

file_path = DATA_DIR / "pre-piloto" / "29_04_2026" / "SujetoNicoCentrodeInnovacion4292026_1.txt"

event_file_path = DATA_DIR / "pre-piloto" / "29_04_2026" / "SujetoNicoCentrodeInnovacion4292026_1_event.txt"

# Dispositivos posibles
POSSIBLE_DEVICES = ["MWMOBILE1", "MWMOBILE2", "MWMOBILE3"]

# Señales esperadas por dispositivo
EXPECTED_SIGNALS = ["ECG", "Resp", "Z0", "EDA"]

# Parámetros ajustables
ECG_BANDPASS = (5, 15)          # Hz
Z0_LOWPASS = 0.7                # Hz
RESP_BAND = (0.05, 0.7)         # Hz
EDA_SMOOTH_WINDOW_SEC = 0.5
EDA_MIN_PROMINENCE = None

SHOW_EVENT_LABELS = True
EVENT_ALPHA = 0.7
EVENT_COLOR = "purple"
EVENT_LINESTYLE = "--"

# Si quieres guardar figuras en vez de solo mostrarlas
SAVE_FIGURES = False

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


def save_or_show(fig, filename):
    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def load_events(event_file_path):
    event_path = Path(event_file_path)

    if not event_path.exists():
        print(f"\nNo se encontró archivo de eventos: {event_file_path}")
        return pd.DataFrame(columns=["Event Type", "Name", "Time"])

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

    if events_df is None or len(events_df) == 0:
        return

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


def detect_available_devices(df):
    """
    Detecta automáticamente los dispositivos que tienen al menos una señal.
    Idealmente deben tener ECG, Resp, Z0 y EDA.
    """

    available_devices = []

    for device in POSSIBLE_DEVICES:
        cols_device = [f"{device}_{sig}" for sig in EXPECTED_SIGNALS]
        existing_cols = [col for col in cols_device if col in df.columns]

        if len(existing_cols) > 0:
            available_devices.append(device)

    return available_devices


def get_device_columns(df, device):
    """
    Retorna las columnas disponibles para un dispositivo específico.
    """

    cols = {}

    for sig in EXPECTED_SIGNALS:
        col = f"{device}_{sig}"
        if col in df.columns:
            cols[sig] = col
        else:
            cols[sig] = None

    return cols


# =========================
# ANÁLISIS POR SEÑAL
# =========================

def plot_raw_signals(df, time_s, device, cols, events_df):
    available = [sig for sig, col in cols.items() if col is not None]

    if len(available) == 0:
        return

    fig, axs = plt.subplots(len(available), 1, figsize=(14, 2.8 * len(available)), sharex=True)

    if len(available) == 1:
        axs = [axs]

    for ax, sig in zip(axs, available):
        col = cols[sig]
        ax.plot(time_s, df[col], linewidth=1)
        ax.set_title(f"{device} - {sig}")
        ax.set_ylabel("Amplitud")
        ax.grid(True)

    axs[-1].set_xlabel("Tiempo (s)")

    add_events_to_axes(
        axs,
        events_df,
        show_labels=SHOW_EVENT_LABELS,
        color=EVENT_COLOR,
        linestyle=EVENT_LINESTYLE,
        alpha=EVENT_ALPHA
    )

    plt.tight_layout()
    save_or_show(fig, f"{device}_01_senales_crudas.png")


def analyze_ecg(df, time_s, fs, device, ecg_col, events_df):
    if ecg_col is None:
        print(f"\n[{device}] No hay ECG. Se omite análisis ECG.")
        return None

    ecg = df[ecg_col].to_numpy()

    ecg_filt = butter_filter(ecg, fs, ECG_BANDPASS, btype='bandpass', order=3)

    distance_samples_ecg = int(0.5 * fs)
    prom_ecg = 1* np.std(ecg_filt)

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

    print(f"\n===== {device}: HRV desde ECG =====")
    print(f"Número de R-peaks detectados: {len(r_peaks)}")
    print(f"Frecuencia cardiaca media: {mean_hr:.2f} bpm")
    print(f"RR medio: {mean_rr * 1000:.2f} ms")
    print(f"SDNN: {sdnn:.2f} ms")
    print(f"RMSSD: {rmssd:.2f} ms")
    print(f"pNN50: {pnn50:.2f} %")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_s, ecg, label="ECG crudo", alpha=0.4)
    ax.plot(time_s, ecg_filt, label="ECG filtrado", linewidth=1.2)
    ax.plot(r_times, ecg_filt[r_peaks], "ro", label="R-peaks")
    ax.set_title(f"{device} - ECG con detección de R-peaks")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    ax.legend()

    add_events_to_axes(
        ax,
        events_df,
        show_labels=SHOW_EVENT_LABELS,
        color=EVENT_COLOR,
        linestyle=EVENT_LINESTYLE,
        alpha=EVENT_ALPHA
    )

    plt.tight_layout()
    save_or_show(fig, f"{device}_02_ecg_rpeaks.png")

    if len(rr) > 0:
        fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

        axs[0].plot(r_times[1:], rr * 1000, marker='o')
        axs[0].set_title(f"{device} - Intervalos RR")
        axs[0].set_xlabel("Tiempo (s)")
        axs[0].set_ylabel("RR (ms)")
        axs[0].grid(True)

        axs[1].plot(r_times[1:], hr_inst, marker='o')
        axs[1].set_title(f"{device} - Frecuencia cardíaca instantánea")
        axs[1].set_xlabel("Tiempo (s)")
        axs[1].set_ylabel("HR (bpm)")
        axs[1].grid(True)

        add_events_to_axes(
            axs,
            events_df,
            show_labels=SHOW_EVENT_LABELS,
            color=EVENT_COLOR,
            linestyle=EVENT_LINESTYLE,
            alpha=EVENT_ALPHA
        )

        plt.tight_layout()
        save_or_show(fig, f"{device}_03_hrv_rr_hr.png")

    return {
        "device": device,
        "n_r_peaks": len(r_peaks),
        "mean_hr_bpm": mean_hr,
        "mean_rr_ms": mean_rr * 1000,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_percent": pnn50
    }


def analyze_z0(df, time_s, fs, device, z0_col, events_df):
    if z0_col is None:
        print(f"\n[{device}] No hay Z0. Se omite análisis Z0.")
        return None

    z0 = df[z0_col].to_numpy()

    z0_centered = z0 - np.mean(z0)
    freqs_z0 = rfftfreq(len(z0_centered), d=1 / fs)
    fft_z0 = np.abs(rfft(z0_centered))

    z0_low = butter_filter(z0, fs, Z0_LOWPASS, btype='low', order=4)

    fig, axs = plt.subplots(3, 1, figsize=(14, 10))

    axs[0].plot(time_s, z0, label="Z0 cruda")
    axs[0].set_title(f"{device} - Z0 cruda")
    axs[0].set_xlabel("Tiempo (s)")
    axs[0].set_ylabel("Amplitud")
    axs[0].grid(True)

    axs[1].plot(freqs_z0, fft_z0)
    axs[1].set_title(f"{device} - FFT de Z0")
    axs[1].set_xlabel("Frecuencia (Hz)")
    axs[1].set_ylabel("Magnitud")
    axs[1].set_xlim(0, min(5, fs / 2))
    axs[1].grid(True)

    axs[2].plot(time_s, z0, alpha=0.4, label="Z0 cruda")
    axs[2].plot(time_s, z0_low, linewidth=2, label=f"Z0 LPF {Z0_LOWPASS} Hz")
    axs[2].set_title(f"{device} - Z0 filtrada pasa-bajos")
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].set_ylabel("Amplitud")
    axs[2].grid(True)
    axs[2].legend()

    add_events_to_axes(
        [axs[0], axs[2]],
        events_df,
        show_labels=SHOW_EVENT_LABELS,
        color=EVENT_COLOR,
        linestyle=EVENT_LINESTYLE,
        alpha=EVENT_ALPHA
    )

    plt.tight_layout()
    save_or_show(fig, f"{device}_04_z0_fft_lpf.png")

    print(f"\n===== {device}: Z0 =====")
    print(f"Media Z0: {np.mean(z0):.4f}")
    print(f"STD Z0: {np.std(z0):.4f}")

    return {
        "device": device,
        "z0_mean": np.mean(z0),
        "z0_std": np.std(z0)
    }


def analyze_resp(df, time_s, fs, device, resp_col, events_df):
    if resp_col is None:
        print(f"\n[{device}] No hay Resp. Se omite análisis respiratorio.")
        return None

    resp = df[resp_col].to_numpy()

    resp_filt = butter_filter(resp, fs, RESP_BAND, btype='bandpass', order=3)
    resp_filt = resp_filt - np.mean(resp_filt)

    # Detección de peaks respiratorios
    min_breath_distance = int(1.2 * fs)
    prominence_resp = 0.2 * np.std(resp_filt)

    resp_peaks, _ = find_peaks(
        resp_filt,
        distance=min_breath_distance,
        prominence=prominence_resp
    )

    resp_mins, _ = find_peaks(
        -resp_filt,
        distance=min_breath_distance,
        prominence=prominence_resp
    )

    resp_peak_times = time_s[resp_peaks]

    if len(resp_peak_times) >= 2:
        breath_intervals = np.diff(resp_peak_times)
        resp_rate_inst = 60.0 / breath_intervals
        mean_resp_rate = np.mean(resp_rate_inst)
    else:
        resp_rate_inst = np.array([])
        mean_resp_rate = np.nan

    print(f"\n===== {device}: frecuencia respiratoria desde Resp =====")
    print(f"Peaks respiratorios detectados: {len(resp_peaks)}")
    print(f"Frecuencia respiratoria media: {mean_resp_rate:.2f} resp/min")

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(time_s, resp, alpha=0.4, label="Resp cruda")
    axs[0].plot(time_s, resp_filt, linewidth=1.5, label=f"Resp filtrada {RESP_BAND[0]}-{RESP_BAND[1]} Hz")
    axs[0].plot(time_s[resp_peaks], resp_filt[resp_peaks], "ro", label="Peaks")
    axs[0].plot(time_s[resp_mins], resp_filt[resp_mins], "go", label="Mínimos")
    axs[0].set_title(f"{device} - Señal respiratoria")
    axs[0].set_ylabel("Amplitud")
    axs[0].grid(True)
    axs[0].legend()

    if len(resp_rate_inst) > 0:
        axs[1].plot(resp_peak_times[1:], resp_rate_inst, marker="o")
    axs[1].set_title(f"{device} - Frecuencia respiratoria instantánea")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("Resp/min")
    axs[1].grid(True)

    add_events_to_axes(
        axs,
        events_df,
        show_labels=SHOW_EVENT_LABELS,
        color=EVENT_COLOR,
        linestyle=EVENT_LINESTYLE,
        alpha=EVENT_ALPHA
    )

    plt.tight_layout()
    save_or_show(fig, f"{device}_05_resp_rate.png")

    return {
        "device": device,
        "n_resp_peaks": len(resp_peaks),
        "mean_resp_rate": mean_resp_rate
    }


def analyze_eda(df, time_s, fs, device, eda_col, events_df):
    if eda_col is None:
        print(f"\n[{device}] No hay EDA. Se omite análisis EDA.")
        return None

    eda = df[eda_col].to_numpy()

    eda_smooth = moving_average(eda, EDA_SMOOTH_WINDOW_SEC * fs)
    deda = np.gradient(eda_smooth, time_s)

    sign_deda = np.sign(deda)
    sign_change = np.diff(sign_deda)

    max_idx = np.where(sign_change < 0)[0] + 1
    min_idx = np.where(sign_change > 0)[0] + 1

    eda_min_prominence = EDA_MIN_PROMINENCE
    if eda_min_prominence is None:
        eda_min_prominence = 0.1 * np.std(eda_smooth)

    valid_max = []
    for idx in max_idx:
        left = max(0, idx - int(1.0 * fs))
        right = min(len(eda_smooth), idx + int(1.0 * fs))
        local_min = np.min(eda_smooth[left:right])

        if (eda_smooth[idx] - local_min) >= eda_min_prominence:
            valid_max.append(idx)

    valid_min = []
    for idx in min_idx:
        left = max(0, idx - int(1.0 * fs))
        right = min(len(eda_smooth), idx + int(1.0 * fs))
        local_max = np.max(eda_smooth[left:right])

        if (local_max - eda_smooth[idx]) >= eda_min_prominence:
            valid_min.append(idx)

    valid_max = np.array(valid_max, dtype=int)
    valid_min = np.array(valid_min, dtype=int)

    print(f"\n===== {device}: EDA máximos y mínimos =====")
    print(f"Máximos detectados: {len(valid_max)}")
    print(f"Mínimos detectados: {len(valid_min)}")
    print(f"Prominencia mínima usada: {eda_min_prominence:.4f}")

    cut_sec = 0.5

    start_idx = int(cut_sec * fs)
    end_idx = len(time_s) - int(cut_sec * fs)

    if end_idx <= start_idx:
        start_idx = 0
        end_idx = len(time_s) - 1

    mask_time = (time_s >= time_s[start_idx]) & (time_s <= time_s[end_idx])

    valid_max_plot = valid_max[(valid_max >= start_idx) & (valid_max <= end_idx)]
    valid_min_plot = valid_min[(valid_min >= start_idx) & (valid_min <= end_idx)]

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(time_s[mask_time], eda[mask_time], alpha=0.4, label="EDA cruda")
    axs[0].plot(time_s[mask_time], eda_smooth[mask_time], linewidth=2, label="EDA suavizada")
    axs[0].plot(time_s[valid_max_plot], eda_smooth[valid_max_plot], "ro", label="Máximos")
    axs[0].plot(time_s[valid_min_plot], eda_smooth[valid_min_plot], "go", label="Mínimos")

    axs[0].set_title(f"{device} - EDA con máximos y mínimos")
    axs[0].set_ylabel("Amplitud")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time_s[mask_time], deda[mask_time], label="Derivada EDA")
    axs[1].axhline(0, color="k", linestyle="--")
    axs[1].set_title(f"{device} - Derivada de EDA")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("dEDA/dt")
    axs[1].grid(True)

    add_events_to_axes(
        axs,
        events_df,
        show_labels=SHOW_EVENT_LABELS,
        color=EVENT_COLOR,
        linestyle=EVENT_LINESTYLE,
        alpha=EVENT_ALPHA
    )

    plt.tight_layout()
    save_or_show(fig, f"{device}_06_eda.png")

    return {
        "device": device,
        "eda_mean": np.mean(eda),
        "eda_std": np.std(eda),
        "n_eda_max": len(valid_max),
        "n_eda_min": len(valid_min)
    }


# =========================
# CARGA DE DATOS
# =========================

df = pd.read_csv(file_path, sep="\t", skiprows=1)
df.columns = df.columns.str.strip()

print("Columnas encontradas:")
print(df.columns.tolist())

if "Time (s)" not in df.columns:
    raise ValueError("No se encontró la columna 'Time (s)'.")

time_s = df["Time (s)"].to_numpy()
fs = estimate_fs(time_s)

print(f"\nFrecuencia de muestreo estimada: {fs:.2f} Hz")

events_df = load_events(event_file_path)

print("\nEventos encontrados:")
print(events_df)

available_devices = detect_available_devices(df)

if len(available_devices) == 0:
    raise ValueError("No se encontró ningún dispositivo MWMOBILE1, MWMOBILE2 o MWMOBILE3.")

print("\nDispositivos detectados:")
print(available_devices)


# =========================
# LOOP PRINCIPAL POR DISPOSITIVO
# =========================

summary_rows = []

for device in available_devices:
    print("\n" + "=" * 60)
    print(f"ANALIZANDO {device}")
    print("=" * 60)

    cols = get_device_columns(df, device)

    print(f"\nColumnas para {device}:")
    for sig, col in cols.items():
        if col is None:
            print(f"  {sig}: NO encontrada")
        else:
            print(f"  {sig}: {col}")

    # Gráficos base
    plot_raw_signals(df, time_s, device, cols, events_df)

    # ECG
    ecg_summary = analyze_ecg(
        df=df,
        time_s=time_s,
        fs=fs,
        device=device,
        ecg_col=cols["ECG"],
        events_df=events_df
    )

    # Resp
    resp_summary = analyze_resp(
        df=df,
        time_s=time_s,
        fs=fs,
        device=device,
        resp_col=cols["Resp"],
        events_df=events_df
    )

    # Z0
    z0_summary = analyze_z0(
        df=df,
        time_s=time_s,
        fs=fs,
        device=device,
        z0_col=cols["Z0"],
        events_df=events_df
    )

    # EDA
    eda_summary = analyze_eda(
        df=df,
        time_s=time_s,
        fs=fs,
        device=device,
        eda_col=cols["EDA"],
        events_df=events_df
    )

    # Unir resumen del dispositivo
    row = {"device": device}

    for summary in [ecg_summary, resp_summary, z0_summary, eda_summary]:
        if summary is not None:
            row.update(summary)

    summary_rows.append(row)


# =========================
# RESUMEN FINAL
# =========================

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(summary_df)

summary_df.to_csv(OUTPUT_DIR / "resumen_dispositivos.csv", index=False)

print(f"\nResumen guardado en:")
print(OUTPUT_DIR / "resumen_dispositivos.csv")