from pathlib import Path
import os

import cv2
from PIL import Image


# ============================================================
# Configuración
# ============================================================

ANCHO_GIF = 360
FPS_GIF = 4
MAX_SEGUNDOS = 4.0

VIDEOS_EVIDENCIA = [
    # DCSASS - Shoplifting048
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_2.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_3.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_12.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_13.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_4.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_5.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_6.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_7.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_8.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_9.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_10.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_14.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_15.mp4",
    "DCSASS__Shoplifting048_x264__Shoplifting048_x264_17.mp4",

    # DCSASS - Shoplifting006
    "DCSASS__Shoplifting006_x264__Shoplifting006_x264_28.mp4",
    "DCSASS__Shoplifting006_x264__Shoplifting006_x264_29.mp4",
    "DCSASS__Shoplifting006_x264__Shoplifting006_x264_19.mp4",
    "DCSASS__Shoplifting006_x264__Shoplifting006_x264_15.mp4",

    # DCSASS - Shoplifting021
    "DCSASS__Shoplifting021_x264__Shoplifting021_x264_23.mp4",
    "DCSASS__Shoplifting021_x264__Shoplifting021_x264_24.mp4",

    # MNNIT - falsos negativos revisados
    "MNNIT__shoplifting-66__MNNIT_065.mp4",
    "MNNIT__shoplifting-60__MNNIT_059.mp4",
    "MNNIT__shoplifting-13__MNNIT_013.mp4",
    "MNNIT__shoplifting-65__MNNIT_064.mp4",
    "MNNIT__shoplifting-69__MNNIT_068.mp4",
    "MNNIT__shoplifting-68__MNNIT_067.mp4",
    "MNNIT__shoplifting-70__MNNIT_069.mp4",
    "MNNIT__shoplifting-67__MNNIT_066.mp4",

    # MNNIT - falso positivo revisado
    "MNNIT__normal-21__MNNIT_113.mp4",
]

NOMBRE_VIDEO_GENERADO = (
    "Shoplifting048_clips_8_9_10_probabilidades.mp4"
)


# ============================================================
# Localización de rutas
# ============================================================

def find_repo_root():
    """
    Localiza la raíz del repositorio sin depender de una
    ruta personal específica.
    """

    candidates = [
        Path.cwd(),
        *Path.cwd().parents,
    ]

    repo_env = os.environ.get("MNA_REPO_DIR")

    if repo_env:
        candidates.append(
            Path(repo_env)
        )

    candidates.append(
        Path(
            "/opt/workspace/"
            "MNA---Proyecto-Integrador"
        )
    )

    for candidate in candidates:

        data_dir = (
            candidate
            / "Datos"
            / "Preparacion_Datasets"
        )

        if data_dir.is_dir():
            return candidate

    raise FileNotFoundError(
        "No se pudo localizar la raíz del repositorio."
    )


REPO_DIR = find_repo_root()

BASE_DIR = (
    REPO_DIR
    / "Datos"
    / "Preparacion_Datasets"
)

GIF_OUTPUT_DIR = (
    BASE_DIR
    / "evidencias_baseline"
    / "gifs"
)

GIF_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLIPS_DIR = Path(
    os.environ.get(
        "BNEXT_CLIPS_DIR",
        "/opt/workspace/datasets/procesados/clips_4s"
    )
)


# ============================================================
# Generación de GIF
# ============================================================

def crear_gif(
    ruta_video,
    ruta_gif,
    ancho=ANCHO_GIF,
    fps_gif=FPS_GIF,
    max_segundos=MAX_SEGUNDOS,
):
    """
    Genera un GIF reducido a partir de un clip MP4.
    """

    cap = cv2.VideoCapture(
        str(ruta_video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"No fue posible abrir: {ruta_video}"
        )

    fps_video = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps_video <= 0:
        fps_video = 24.0

    salto_frames = max(
        1,
        round(
            fps_video
            / fps_gif
        )
    )

    max_frames_gif = int(
        fps_gif
        * max_segundos
    )

    frames = []
    indice_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if indice_frame % salto_frames == 0:

            alto_original, ancho_original = (
                frame.shape[:2]
            )

            factor = (
                ancho
                / ancho_original
            )

            alto_nuevo = int(
                alto_original
                * factor
            )

            frame = cv2.resize(
                frame,
                (
                    ancho,
                    alto_nuevo
                ),
                interpolation=cv2.INTER_AREA,
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(
                Image.fromarray(
                    frame
                )
            )

            if len(frames) >= max_frames_gif:
                break

        indice_frame += 1

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No se extrajeron frames de {ruta_video.name}"
        )

    duracion_frame_ms = round(
        1000
        / fps_gif
    )

    frames[0].save(
        ruta_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duracion_frame_ms,
        loop=0,
        optimize=True,
    )


# ============================================================
# Ejecución
# ============================================================

def main():

    print(
        "Repositorio:",
        REPO_DIR
    )

    print(
        "Directorio de clips:",
        CLIPS_DIR
    )

    print(
        "Directorio de salida:",
        GIF_OUTPUT_DIR.relative_to(
            REPO_DIR
        )
    )

    print()

    for nombre_video in VIDEOS_EVIDENCIA:

        ruta_video = (
            CLIPS_DIR
            / nombre_video
        )

        if not ruta_video.is_file():
            print(
                "[NO ENCONTRADO]",
                nombre_video
            )
            continue

        nombre_gif = (
            Path(nombre_video)
            .stem
            .split("__")[-1]
            + ".gif"
        )

        ruta_gif = (
            GIF_OUTPUT_DIR
            / nombre_gif
        )

        crear_gif(
            ruta_video=ruta_video,
            ruta_gif=ruta_gif,
        )

        tamano_mb = (
            ruta_gif.stat().st_size
            / (1024 ** 2)
        )

        print(
            "[OK]",
            nombre_gif,
            f"{tamano_mb:.2f} MB"
        )
    # --------------------------------------------------------
    # Video combinado de la transición temporal 8 -> 9 -> 10
    # --------------------------------------------------------

    ruta_video_generado = (
        BASE_DIR
        / "evidencias_baseline"
        / NOMBRE_VIDEO_GENERADO
    )

    if not ruta_video_generado.is_file():

        print(
            "[NO ENCONTRADO]",
            NOMBRE_VIDEO_GENERADO
        )

    else:

        nombre_gif = (
            ruta_video_generado.stem
            + ".gif"
        )

        ruta_gif = (
            GIF_OUTPUT_DIR
            / nombre_gif
        )

        crear_gif(
            ruta_video=ruta_video_generado,
            ruta_gif=ruta_gif,
            ancho=ANCHO_GIF,
            fps_gif=FPS_GIF,
            max_segundos=12.0,
        )

        tamano_mb = (
            ruta_gif.stat().st_size
            / (1024 ** 2)
        )

        print(
            "[OK]",
            nombre_gif,
            f"{tamano_mb:.2f} MB"
        )

if __name__ == "__main__":
    main()