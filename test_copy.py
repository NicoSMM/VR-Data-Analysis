import math
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================
file_path = "3_0.txt"
mode = "grouped"      # "single" o "grouped"
plots_per_figure = 4
sep = "\t"
skiprows = 1

# =========================
# LECTURA
# =========================
df = pd.read_csv(file_path, sep=sep, skiprows=skiprows)
df.columns = df.columns.str.strip()

print("Columnas encontradas:")
print(df.columns.tolist())

# Buscar columna de tiempo
time_candidates = [col for col in df.columns if "time" in col.lower()]
if not time_candidates:
    raise ValueError("No se encontró una columna de tiempo.")

time_col = time_candidates[0]
signal_cols = [col for col in df.columns if col != time_col]

# =========================
# GRAFICAR
# =========================
if mode == "single":
    fig, axs = plt.subplots(len(signal_cols), 1, figsize=(14, 2.8 * len(signal_cols)), sharex=True)

    if len(signal_cols) == 1:
        axs = [axs]

    for ax, signal in zip(axs, signal_cols):
        ax.plot(df[time_col], df[signal])
        ax.set_title(signal)
        ax.set_ylabel("Amplitud")
        ax.grid(True)

    axs[-1].set_xlabel(time_col)
    fig.suptitle(file_path, fontsize=14)
    plt.tight_layout()
    plt.show()

elif mode == "grouped":
    n_figures = math.ceil(len(signal_cols) / plots_per_figure)

    for i in range(n_figures):
        start = i * plots_per_figure
        end = min(start + plots_per_figure, len(signal_cols))
        group = signal_cols[start:end]

        fig, axs = plt.subplots(len(group), 1, figsize=(14, 2.8 * len(group)), sharex=True)

        if len(group) == 1:
            axs = [axs]

        for ax, signal in zip(axs, group):
            ax.plot(df[time_col], df[signal])
            ax.set_title(signal)
            ax.set_ylabel("Amplitud")
            ax.grid(True)

        axs[-1].set_xlabel(time_col)
        fig.suptitle(f"{file_path} | Figura {i+1}", fontsize=14)
        plt.tight_layout()
        plt.show()

else:
    raise ValueError("mode debe ser 'single' o 'grouped'")