#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

YOLO_RESULTS_DIR = RESULTS_DIR / "yolo_pose"
YOLO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURACIÓN
# ============================================================
INPUT_FOLDER = DATA_DIR / "pre-piloto" / "29_04_2026" / "29_04_2026" / "Exp"
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
MODEL_PATH = "yolo11n-pose.pt"
CONFIDENCE_THRESHOLD = 0.4
RECURSIVE = True

# ============================================================
# NOMBRES DE KEYPOINTS DEL MODELO COCO
# ============================================================

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# ============================================================
# FUNCIONES
# ============================================================

def buscar_videos(input_folder, recursive=True):
    """
    Busca todos los videos dentro de una carpeta.

    Parameters
    ----------
    input_folder : Path or str
        Carpeta donde se buscarán videos.
    recursive : bool
        Si True, busca también en subcarpetas.

    Returns
    -------
    list[Path]
        Lista ordenada de videos encontrados.
    """

    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"No se encontró la carpeta: {input_folder}")

    videos = []

    if recursive:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(input_folder.rglob(f"*{ext}"))
            videos.extend(input_folder.rglob(f"*{ext.upper()}"))
    else:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(input_folder.glob(f"*{ext}"))
            videos.extend(input_folder.glob(f"*{ext.upper()}"))

    videos = sorted(set(videos))

    return videos


def crear_carpetas_resultado(video_path):
    """
    Crea las carpetas de resultado para un video específico,
    respetando la estructura relativa dentro de data/.

    Ejemplo:
    video_path:
        data/29_04_2026/Exp/video.mp4

    resultados:
        results/yolo_pose/29_04_2026/Exp/videos/
        results/yolo_pose/29_04_2026/Exp/csv/
    """

    video_path = Path(video_path)

    try:
        relative_folder = video_path.parent.relative_to(DATA_DIR)
    except ValueError:
        # Si el video no está dentro de data/, usa solo el nombre de la carpeta
        relative_folder = Path(video_path.parent.name)

    session_results_dir = YOLO_RESULTS_DIR / relative_folder

    video_results_dir = session_results_dir / "videos"
    csv_results_dir = session_results_dir / "csv"

    video_results_dir.mkdir(parents=True, exist_ok=True)
    csv_results_dir.mkdir(parents=True, exist_ok=True)

    return relative_folder, video_results_dir, csv_results_dir


def procesar_video(model, video_path):
    """
    Procesa un video con YOLO-Pose.

    Guarda:
    1. Video anotado con keypoints.
    2. CSV con coordenadas de keypoints por frame.
    """

    video_path = Path(video_path)

    print("\n" + "=" * 80)
    print(f"Procesando video: {video_path.name}")
    print("=" * 80)

    relative_folder, video_results_dir, csv_results_dir = crear_carpetas_resultado(video_path)

    output_video_path = video_results_dir / f"{video_path.stem}_pose.mp4"
    output_csv_path = csv_results_dir / f"{video_path.stem}_keypoints.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 0:
        print(f"No se pudo leer correctamente el FPS del video: {video_path.name}")
        cap.release()
        return

    if width <= 0 or height <= 0:
        print(f"No se pudo leer correctamente el tamaño del video: {video_path.name}")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        # Guardar frame anotado
        annotated_frame = result.plot()
        out.write(annotated_frame)

        # Guardar keypoints en CSV
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()

            for person_id, person_kpts in enumerate(keypoints_xy):
                row = {
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "session_folder": str(relative_folder),
                    "frame": frame_idx,
                    "time_s": frame_idx / fps,
                    "person_id": person_id,
                }

                for i, name in enumerate(KEYPOINT_NAMES):
                    row[f"{name}_x"] = person_kpts[i, 0]
                    row[f"{name}_y"] = person_kpts[i, 1]
                    row[f"{name}_conf"] = keypoints_conf[person_id, i]

                rows.append(row)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    print(f"Video guardado en: {output_video_path}")
    print(f"CSV guardado en: {output_csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Buscando videos...")
    print(f"Carpeta de entrada: {INPUT_FOLDER}")

    videos = buscar_videos(INPUT_FOLDER, recursive=RECURSIVE)

    if len(videos) == 0:
        print("No se encontraron videos en la carpeta indicada.")
        return

    print(f"\nSe encontraron {len(videos)} videos:")

    for video in videos:
        print(f"- {video.name}")

    print("\nCargando modelo YOLO-Pose...")
    model = YOLO(MODEL_PATH)

    for video in videos:
        procesar_video(model, video)

    print("\nProceso completo.")
    print(f"Resultados guardados en: {YOLO_RESULTS_DIR}")


if __name__ == "__main__":
    main()