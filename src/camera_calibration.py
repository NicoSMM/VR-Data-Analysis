#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibración manual de cámaras/videos con detección de segmentos estables.

Este script:
1. Busca videos dentro de una carpeta madre.
2. Analiza movimiento global de cámara.
3. Detecta segmentos donde la cámara parece estable.
4. Permite calibrar un frame representativo por cada segmento estable.
5. Permite hacer clic en puntos conocidos de la imagen.
6. Pide por terminal las coordenadas 3D reales de cada punto.
7. Guarda los puntos 2D-3D en un archivo JSON por sesión.
8. Calcula una calibración aproximada con solvePnP.

Estructura esperada:

proyecto/
├── src/
│   └── calibrar_camaras_manual.py
├── data/
│   └── pre-piloto/
│       └── 29_04_2026/
│           └── 29_04_2026/
│               └── Exp/
│                   ├── Setup Video 1 ...
│                   ├── Setup Video 3 ...
│                   └── ...
└── results/
    └── calibration/
        └── pre-piloto/
            └── 29_04_2026/
                └── 29_04_2026/
                    └── Exp/
                        └── calibracion_manual.json
"""

from pathlib import Path
import cv2
import numpy as np
import json


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


# ============================================================
# CARPETA DE CALIBRACIÓN POR SESIÓN
# ============================================================

def crear_ruta_calibracion_sesion(input_folder):
    """
    Crea una carpeta de calibración dentro de results/calibration/
    respetando la estructura relativa de la sesión dentro de data/.
    """

    input_folder = Path(input_folder).resolve()

    try:
        relative_session_folder = input_folder.relative_to(DATA_DIR.resolve())
    except ValueError:
        relative_session_folder = Path(input_folder.name)

    calibration_session_dir = RESULTS_DIR / "calibration" / relative_session_folder
    calibration_session_dir.mkdir(parents=True, exist_ok=True)

    output_json = calibration_session_dir / "calibracion_manual.json"

    return relative_session_folder, calibration_session_dir, output_json


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Carpeta madre donde están los videos
INPUT_FOLDER = DATA_DIR / "pre-piloto" / "29_04_2026" / "29_04_2026" / "Exp"

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# Frame fijo alternativo si no se detectan segmentos estables
FRAME_INDEX = 0

# Cantidad mínima recomendada de puntos 2D-3D para solvePnP.
MIN_POINTS_FOR_PNP = 6

# Parámetros de detección de movimiento de cámara
SAMPLE_EVERY_SECONDS = 0.5       # cada cuánto revisar movimiento
MOTION_THRESHOLD_PX = 8.0        # umbral de movimiento en píxeles
MIN_STABLE_SECONDS = 2.0         # duración mínima para considerar segmento estable

# Crear carpeta de calibración específica para esta sesión
RELATIVE_SESSION_FOLDER, CALIBRATION_DIR, OUTPUT_JSON = crear_ruta_calibracion_sesion(INPUT_FOLDER)


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def buscar_videos(input_folder):
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"No existe la carpeta: {input_folder}")

    videos = []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_folder.rglob(f"*{ext}"))
        videos.extend(input_folder.rglob(f"*{ext.upper()}"))

    return sorted(set(videos))


def leer_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_index = min(frame_index, total_frames - 1)
    frame_index = max(frame_index, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"No se pudo leer el frame {frame_index} de {video_path}")

    info = {
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_index": frame_index,
    }

    return frame, info


def guardar_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nCalibración guardada en:")
    print(output_path)


# ============================================================
# DETECCIÓN DE MOVIMIENTO DE CÁMARA
# ============================================================

def frame_a_gray(frame, resize_width=640):
    """
    Convierte un frame a escala de grises y lo reduce para acelerar el análisis.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    if w > resize_width:
        scale = resize_width / w
        new_h = int(h * scale)
        gray = cv2.resize(gray, (resize_width, new_h))

    return gray


def calcular_motion_score_orb(gray_prev, gray_curr):
    """
    Calcula un puntaje de movimiento global entre dos frames usando ORB.

    Retorna:
    - motion_score: desplazamiento mediano de puntos coincidentes en píxeles.
    - n_matches: número de matches usados.
    """

    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return None, len(matches)

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[: min(100, len(matches))]

    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    desplazamientos = np.linalg.norm(pts2 - pts1, axis=1)

    motion_score = float(np.median(desplazamientos))

    return motion_score, len(matches)


def analizar_estabilidad_video(
    video_path,
    sample_every_seconds=0.5,
    motion_threshold_px=8.0,
    min_stable_seconds=2.0,
):
    """
    Analiza un video y detecta segmentos donde la cámara parece estable.

    Retorna:
    - segmentos: lista de segmentos estables.
    - diagnostics: información completa del análisis de movimiento.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"No se pudo leer FPS del video: {video_path}")

    step_frames = max(1, int(round(sample_every_seconds * fps)))
    min_stable_frames = int(round(min_stable_seconds * fps))

    sampled = []

    prev_gray = None
    frame_indices = list(range(0, total_frames, step_frames))

    print(f"Analizando movimiento cada {step_frames} frames aprox.")

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        gray = frame_a_gray(frame)

        if prev_gray is None:
            sampled.append(
                {
                    "frame": int(frame_idx),
                    "time_s": float(frame_idx / fps),
                    "motion_score": 0.0,
                    "stable": True,
                    "n_matches": 0,
                }
            )
        else:
            motion_score, n_matches = calcular_motion_score_orb(prev_gray, gray)

            if motion_score is None:
                stable = False
                motion_score_value = None
            else:
                stable = motion_score <= motion_threshold_px
                motion_score_value = float(motion_score)

            sampled.append(
                {
                    "frame": int(frame_idx),
                    "time_s": float(frame_idx / fps),
                    "motion_score": motion_score_value,
                    "stable": bool(stable),
                    "n_matches": int(n_matches),
                }
            )

        prev_gray = gray

        if idx % 20 == 0:
            print(f"Analizando muestra {idx + 1}/{len(frame_indices)}")

    cap.release()

    segmentos = []
    in_segment = False
    start_sample = None

    for i, item in enumerate(sampled):
        is_stable = item["stable"]

        if is_stable and not in_segment:
            in_segment = True
            start_sample = i

        is_last = i == len(sampled) - 1

        if in_segment and ((not is_stable) or is_last):
            end_sample = i - 1 if not is_stable else i

            start_frame = sampled[start_sample]["frame"]
            end_frame = sampled[end_sample]["frame"]

            duration_frames = end_frame - start_frame

            if duration_frames >= min_stable_frames:
                segment_items = sampled[start_sample: end_sample + 1]

                valid_scores = [
                    x["motion_score"]
                    for x in segment_items
                    if x["motion_score"] is not None
                ]

                if len(valid_scores) > 0:
                    mean_motion_score = float(np.mean(valid_scores))
                    median_motion_score = float(np.median(valid_scores))
                else:
                    mean_motion_score = None
                    median_motion_score = None

                representative_frame = int((start_frame + end_frame) / 2)

                segmentos.append(
                    {
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "representative_frame": int(representative_frame),
                        "start_time_s": float(start_frame / fps),
                        "end_time_s": float(end_frame / fps),
                        "duration_s": float((end_frame - start_frame) / fps),
                        "mean_motion_score": mean_motion_score,
                        "median_motion_score": median_motion_score,
                    }
                )

            in_segment = False
            start_sample = None

    diagnostics = {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "sample_every_seconds": float(sample_every_seconds),
        "motion_threshold_px": float(motion_threshold_px),
        "min_stable_seconds": float(min_stable_seconds),
        "sampled_motion": sampled,
    }

    return segmentos, diagnostics


# ============================================================
# GUI PARA SELECCIONAR PUNTOS
# ============================================================

class PointSelector:
    def __init__(self, image, window_name="Seleccionar puntos"):
        self.original = image.copy()
        self.image = image.copy()
        self.window_name = window_name
        self.points_2d = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points_2d.append([float(x), float(y)])
            self.redibujar()

    def redibujar(self):
        self.image = self.original.copy()

        for i, (x, y) in enumerate(self.points_2d):
            x_int = int(round(x))
            y_int = int(round(y))

            cv2.circle(self.image, (x_int, y_int), 6, (0, 0, 255), -1)
            cv2.putText(
                self.image,
                str(i + 1),
                (x_int + 8, y_int - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, self.image)

    def seleccionar(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\nInstrucciones:")
        print("- Click izquierdo: agregar punto.")
        print("- Tecla 'u': deshacer último punto.")
        print("- Tecla 'r': reiniciar puntos.")
        print("- Tecla 'q': terminar selección de este frame.")
        print("- Tecla 'esc': cancelar.")

        self.redibujar()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord("u"):
                if len(self.points_2d) > 0:
                    self.points_2d.pop()
                    self.redibujar()

            elif key == ord("r"):
                self.points_2d = []
                self.redibujar()

            elif key == ord("q"):
                break

            elif key == 27:
                raise KeyboardInterrupt("Calibración cancelada por el usuario.")

        cv2.destroyWindow(self.window_name)

        return self.points_2d


def pedir_puntos_3d(points_2d):
    """
    Pide las coordenadas 3D reales de cada punto marcado.
    """

    points_3d = []

    print("\nAhora ingresa las coordenadas reales 3D de cada punto.")
    print("Usa el sistema de referencia que tú definas.")
    print("Ejemplo: X Y Z en metros, separado por espacios.")
    print("Ejemplo: 1.20 0.00 0.75")

    for i, (x, y) in enumerate(points_2d):
        while True:
            entrada = input(
                f"Punto {i + 1} | pixel ({x:.1f}, {y:.1f}) | Ingresa X Y Z: "
            ).strip()

            try:
                X, Y, Z = map(float, entrada.replace(",", ".").split())
                points_3d.append([X, Y, Z])
                break
            except ValueError:
                print("Formato inválido. Escribe algo como: 1.20 0.00 0.75")

    return points_3d


# ============================================================
# CALIBRACIÓN PNP
# ============================================================

def estimar_matriz_camara_aproximada(width, height):
    """
    Matriz intrínseca aproximada.

    Esto sirve para partir, pero lo ideal es calibrar intrínsecos
    con checkerboard o Charuco.
    """

    fx = max(width, height)
    fy = max(width, height)
    cx = width / 2
    cy = height / 2

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    dist = np.zeros((5, 1), dtype=np.float64)

    return K, dist


def calcular_pnp(points_2d, points_3d, width, height):
    """
    Calcula pose de cámara usando solvePnP.

    Retorna:
    - K: matriz intrínseca
    - dist: coeficientes de distorsión
    - rvec: vector de rotación
    - tvec: vector de traslación
    - R: matriz de rotación
    - P: matriz de proyección K [R|t]
    - reprojection_error_px: error medio de reproyección en píxeles
    """

    image_points = np.array(points_2d, dtype=np.float64)
    object_points = np.array(points_3d, dtype=np.float64)

    K, dist = estimar_matriz_camara_aproximada(width, height)

    if len(points_2d) < MIN_POINTS_FOR_PNP:
        print(
            f"No se calculará solvePnP porque hay solo {len(points_2d)} puntos. "
            f"Se recomiendan al menos {MIN_POINTS_FOR_PNP}."
        )
        return None

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        print("solvePnP no pudo estimar la pose.")
        return None

    R, _ = cv2.Rodrigues(rvec)

    Rt = np.hstack([R, tvec])
    P = K @ Rt

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist,
    )

    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(projected_points - image_points, axis=1)
    reprojection_error_px = float(np.mean(errors))

    result = {
        "K": K.tolist(),
        "dist": dist.tolist(),
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "R": R.tolist(),
        "P": P.tolist(),
        "reprojection_error_px": reprojection_error_px,
    }

    return result


def calibrar_video(video_path, frame_index, segment_info=None):
    """
    Calibra un video usando un frame específico.
    """

    print("\n" + "=" * 80)
    print(f"Calibrando video:")
    print(video_path)
    print(f"Frame de calibración: {frame_index}")
    print("=" * 80)

    frame, info = leer_frame(video_path, frame_index)

    selector = PointSelector(
        frame,
        window_name=f"Calibracion: {video_path.name} | frame {frame_index}",
    )

    points_2d = selector.seleccionar()

    if len(points_2d) == 0:
        print("No se seleccionaron puntos.")
        return None

    points_3d = pedir_puntos_3d(points_2d)

    pnp_result = calcular_pnp(
        points_2d=points_2d,
        points_3d=points_3d,
        width=info["width"],
        height=info["height"],
    )

    try:
        relative_path = str(video_path.relative_to(DATA_DIR))
    except ValueError:
        relative_path = str(video_path)

    calibration_data = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "relative_path": relative_path,
        "frame_used_for_calibration": int(info["frame_index"]),
        "segment_info": segment_info,
        "video_info": info,
        "points_2d": points_2d,
        "points_3d": points_3d,
        "pnp": pnp_result,
    }

    return calibration_data


# ============================================================
# MAIN
# ============================================================

def main():
    print("Buscando videos...")
    print(f"Carpeta: {INPUT_FOLDER}")
    print(f"Resultados de calibración en: {OUTPUT_JSON}")

    videos = buscar_videos(INPUT_FOLDER)

    if len(videos) == 0:
        print("No se encontraron videos.")
        return

    print(f"\nSe encontraron {len(videos)} videos:")
    for i, video in enumerate(videos):
        print(f"{i + 1}. {video.name}")

    calibraciones = []
    motion_analysis = []

    for video_path in videos:
        respuesta = input(
            f"\n¿Analizar/calibrar este video? {video_path.name} [s/n]: "
        ).strip().lower()

        if respuesta not in ["s", "si", "sí", "y", "yes"]:
            print("Saltando video.")
            continue

        print("\nAnalizando estabilidad de cámara...")

        stable_segments, motion_diagnostics = analizar_estabilidad_video(
            video_path,
            sample_every_seconds=SAMPLE_EVERY_SECONDS,
            motion_threshold_px=MOTION_THRESHOLD_PX,
            min_stable_seconds=MIN_STABLE_SECONDS,
        )

        motion_record = {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "stable_segments": stable_segments,
            "motion_diagnostics": motion_diagnostics,
        }

        motion_analysis.append(motion_record)

        if len(stable_segments) == 0:
            print("\nNo se detectaron segmentos estables claros.")
            print("Puedes calibrar manualmente usando FRAME_INDEX fijo.")

            respuesta_manual = input(
                f"¿Calibrar usando FRAME_INDEX = {FRAME_INDEX}? [s/n]: "
            ).strip().lower()

            if respuesta_manual in ["s", "si", "sí", "y", "yes"]:
                calibration_data = calibrar_video(
                    video_path,
                    frame_index=FRAME_INDEX,
                    segment_info={
                        "type": "manual_fixed_frame",
                        "start_frame": None,
                        "end_frame": None,
                        "representative_frame": int(FRAME_INDEX),
                        "note": "No stable segment detected. Manual fixed frame used.",
                    },
                )

                if calibration_data is not None:
                    calibraciones.append(calibration_data)

        else:
            print(f"\nSe detectaron {len(stable_segments)} segmentos estables:")

            for idx, seg in enumerate(stable_segments):
                print(
                    f"{idx + 1}. "
                    f"{seg['start_time_s']:.2f}s - {seg['end_time_s']:.2f}s | "
                    f"frames {seg['start_frame']} - {seg['end_frame']} | "
                    f"frame representativo: {seg['representative_frame']} | "
                    f"duración: {seg['duration_s']:.2f}s | "
                    f"motion medio: {seg['mean_motion_score']}"
                )

            for idx, seg in enumerate(stable_segments):
                respuesta_seg = input(
                    f"\n¿Calibrar segmento estable {idx + 1}/{len(stable_segments)} "
                    f"del video {video_path.name}? [s/n]: "
                ).strip().lower()

                if respuesta_seg not in ["s", "si", "sí", "y", "yes"]:
                    print("Saltando segmento.")
                    continue

                calibration_data = calibrar_video(
                    video_path,
                    frame_index=seg["representative_frame"],
                    segment_info={
                        "type": "stable_segment",
                        "segment_index": int(idx),
                        **seg,
                    },
                )

                if calibration_data is not None:
                    calibraciones.append(calibration_data)

        guardar_json(
            {
                "input_folder": str(INPUT_FOLDER),
                "relative_session_folder": str(RELATIVE_SESSION_FOLDER),
                "calibration_dir": str(CALIBRATION_DIR),
                "calibrations": calibraciones,
                "motion_analysis": motion_analysis,
            },
            OUTPUT_JSON,
        )

    guardar_json(
        {
            "input_folder": str(INPUT_FOLDER),
            "relative_session_folder": str(RELATIVE_SESSION_FOLDER),
            "calibration_dir": str(CALIBRATION_DIR),
            "calibrations": calibraciones,
            "motion_analysis": motion_analysis,
        },
        OUTPUT_JSON,
    )

    print("\nProceso de calibración terminado.")


if __name__ == "__main__":
    main()