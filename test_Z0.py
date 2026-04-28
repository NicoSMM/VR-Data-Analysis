import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

file_path = r"27_02_2026\Live Data Recording 3_0.txt"

# Buscar sample rate y línea donde empieza la tabla
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

fs = None
header_line = None

for i, line in enumerate(lines):
    if "Sample Rate" in line:
        fs = float(line.split()[-1])
    if line.startswith("Time"):
        header_line = i

print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Encabezado encontrado en línea: {header_line}")

df = pd.read_csv(
    file_path,
    sep="\t",
    skiprows=header_line
)

# Renombrar columnas para trabajar más fácil
df.columns = ["Time", "ECG", "Z0", "dZdt", "EDA"]

t = df["Time"].values
ecg = df["ECG"].values
z0 = df["Z0"].values
dzdt = df["dZdt"].values
eda = df["EDA"].values

# =========================
# FUNCIONES AUXILIARES
# =========================

def plot_signal(t, signal, title, ylabel):
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Tiempo [s]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_rr_hr(r_peaks, fs):
    rr = np.diff(r_peaks) / fs
    hr = 60 / rr
    time_rr = r_peaks[1:] / fs
    return time_rr, rr, hr


# =========================
# 1. GRAFICAR ECG
# =========================

plot_signal(t, ecg, "Señal ECG", "ECG [V]")

# =========================
# 2. DETECCIÓN R POR MÁXIMOS
# =========================

# Distancia mínima entre R peaks
# 0.4 s equivale a máximo aprox. 150 bpm
min_distance = int(0.4 * fs)

# Umbral simple basado en prominencia
peaks_max, properties = find_peaks(
    ecg,
    distance=min_distance,
    prominence=np.std(ecg) * 0.5
)

time_rr_max, rr_max, hr_max = compute_rr_hr(peaks_max, fs)

print("\n=== Método por máximos ===")
print(f"Número de peaks R detectados: {len(peaks_max)}")
print(f"RR medio: {np.mean(rr_max):.3f} s")
print(f"FC media: {np.mean(hr_max):.2f} bpm")

plt.figure(figsize=(12, 4))
plt.plot(t, ecg, label="ECG")
plt.plot(t[peaks_max], ecg[peaks_max], "ro", label="R detectados")
plt.title("ECG con peaks R detectados por máximos")
plt.xlabel("Tiempo [s]")
plt.ylabel("ECG [V]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 3. PAN-TOMPKINS SIMPLIFICADO
# =========================

def bandpass_filter(signal, fs, lowcut=5, highcut=15, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# Paso 1: filtro pasabanda
ecg_filt = bandpass_filter(ecg, fs)

# Paso 2: derivada
ecg_derivative = np.gradient(ecg_filt)

# Paso 3: cuadrado
ecg_squared = ecg_derivative ** 2

# Paso 4: integración por ventana móvil
window_size = int(0.15 * fs)
window = np.ones(window_size) / window_size
ecg_integrated = np.convolve(ecg_squared, window, mode="same")

# Detectar peaks sobre señal integrada
peaks_pt_integrated, _ = find_peaks(
    ecg_integrated,
    distance=min_distance,
    prominence=np.std(ecg_integrated) * 0.5
)

# Refinar posición del peak buscando máximo local en ECG original
search_window = int(0.08 * fs)
peaks_pt = []

for p in peaks_pt_integrated:
    start = max(p - search_window, 0)
    end = min(p + search_window, len(ecg))
    local_max = start + np.argmax(ecg[start:end])
    peaks_pt.append(local_max)

peaks_pt = np.array(sorted(set(peaks_pt)))

time_rr_pt, rr_pt, hr_pt = compute_rr_hr(peaks_pt, fs)

print("\n=== Pan-Tompkins simplificado ===")
print(f"Número de peaks R detectados: {len(peaks_pt)}")
print(f"RR medio: {np.mean(rr_pt):.3f} s")
print(f"FC media: {np.mean(hr_pt):.2f} bpm")

plt.figure(figsize=(12, 4))
plt.plot(t, ecg, label="ECG")
plt.plot(t[peaks_pt], ecg[peaks_pt], "ro", label="R Pan-Tompkins")
plt.title("ECG con peaks R detectados por Pan-Tompkins simplificado")
plt.xlabel("Tiempo [s]")
plt.ylabel("ECG [V]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 4. VARIACIÓN RR Y FC
# =========================

plt.figure(figsize=(12, 4))
plt.plot(time_rr_max, rr_max, label="RR por máximos")
plt.plot(time_rr_pt, rr_pt, label="RR Pan-Tompkins")
plt.title("Variación del intervalo R-R")
plt.xlabel("Tiempo [s]")
plt.ylabel("Intervalo R-R [s]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time_rr_max, hr_max, label="FC por máximos")
plt.plot(time_rr_pt, hr_pt, label="FC Pan-Tompkins")
plt.title("Variación de la frecuencia cardíaca")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia cardíaca [bpm]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 5. GRAFICAR Z0 Y dZ/dt
# =========================

plot_signal(t, z0, "Señal Z0", "Z0 [Ohm]")
plot_signal(t, dzdt, "Señal dZ/dt", "dZ/dt")

# =========================
# 6. DERIVADA DE Z0
# =========================

dz0_dt = - np.gradient(z0, t)

plt.figure(figsize=(12, 4))
plt.plot(t, dz0_dt, label="Derivada calculada de Z0")
plt.plot(t, dzdt, label="dZ/dt medido", alpha=0.8)
plt.title("Comparación entre derivada de Z0 y dZ/dt")
plt.xlabel("Tiempo [s]")
plt.ylabel("dZ/dt")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#6.1 Filtrado de la derivada de Z0 para reducir ruido
b, a = butter(N=4, Wn=0.1, btype = 'low')  # Filtro pasa bajos con frecuencia de corte a 0.1*Nyquist
dz0_dt_filt = filtfilt(b, a, dz0_dt)
plt.figure(figsize=(12, 4))
plt.plot(t, dz0_dt_filt, label="Derivada de Z0 filtrada")
plt.plot(t, dz0_dt, label="Derivada calculada de Z0")
plt.plot(t, dzdt, label="dZ/dt medido", alpha=0.8)
plt.title("Comparación entre derivada de Z0 filtrada y dZ/dt")
plt.xlabel("Tiempo [s]")
plt.ylabel("dZ/dt")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#6.2 Promedio movil de la derivada de Z0 para suavizar la señal
window_size = int(0.05 * fs)  # Ventana de 50 ms
dz0_dt_smooth = np.convolve(dz0_dt, np.ones(window_size)/window_size, mode='same')

plt.figure(figsize=(12, 4))
plt.plot(t, dz0_dt_smooth, label="Derivada de Z0 suavizada")
plt.plot(t, dz0_dt, label="Derivada calculada de Z0")
plt.plot(t, dz0_dt_filt, label="Derivada de Z0 filtrada")
plt.plot(t, dzdt, label="dZ/dt medido", alpha=0.8)
plt.title("Comparación entre derivada de Z0 suavizada y dZ/dt")
plt.xlabel("Tiempo [s]")
plt.ylabel("dZ/dt")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# =========================
# 7. ERROR ENTRE DERIVADA DE Z0 Y dZ/dt
# =========================

error = dz0_dt - dzdt
abs_error = np.abs(error)

error_acumulado = np.sum(abs_error)
error_medio = np.mean(abs_error)
rmse = np.sqrt(np.mean(error ** 2))

print("\n=== Error entre derivada de Z0 y dZ/dt ===")
print(f"Error acumulado absoluto: {error_acumulado:.6f}")
print(f"Error medio absoluto: {error_medio:.6f}")
print(f"RMSE: {rmse:.6f}")


error2 = dz0_dt_filt - dzdt
abs_error2 = np.abs(error2)

error_acumulado2 = np.sum(abs_error2)
error_medio2 = np.mean(abs_error2)
rmse2 = np.sqrt(np.mean(error2 ** 2))

print("\n=== Error entre derivada de Z0 Filtrada y dZ/dt ===")
print(f"Error acumulado absoluto: {error_acumulado2:.6f}")
print(f"Error medio absoluto: {error_medio2:.6f}")
print(f"RMSE: {rmse2:.6f}")


error3 = dz0_dt_smooth - dzdt
abs_error3 = np.abs(error3)

error_acumulado3 = np.sum(abs_error3)
error_medio3 = np.mean(abs_error3)
rmse3 = np.sqrt(np.mean(error3 ** 2))

print("\n=== Error entre derivada de Z0 Suavizada y dZ/dt ===")
print(f"Error acumulado absoluto: {error_acumulado3:.6f}")
print(f"Error medio absoluto: {error_medio3:.6f}")
print(f"RMSE: {rmse3:.6f}")

# =========================
# 8. GRAFICAR EDA
# =========================

plot_signal(t, eda, "Señal EDA", "EDA")