from pathlib import Path
import cv2
import pandas as pd


# Ruta a los datasets en el servidor BNext.
DATASETS_ROOT = Path("/opt/workspace/datasets")

# Ruta de salida del CSV dentro del repositorio.
OUTPUT_CSV = (
    Path(__file__).resolve().parents[1]
    / "videos_metadata.csv"
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def inferir_fuente(video_path: Path) -> str:
    """
    Determina la fuente a partir de la carpeta principal
    dentro de /opt/workspace/datasets.
    """
    try:
        relative = video_path.relative_to(DATASETS_ROOT)
        return relative.parts[0]
    except ValueError:
        return "desconocida"


def inferir_clase(video_path: Path) -> str:
    """
    Infiere la clase solo cuando la estructura de carpetas
    proporciona una etiqueta inequívoca.

    En caso contrario, devuelve 'pendiente'.
    """
    partes = [parte.lower() for parte in video_path.parts]

    if "shoplifting" in partes:
        return "ocultamiento"

    if "normal" in partes:
        return "sin_ocultamiento"

    return "pendiente"

def determinar_estado_muestra(video_path: Path) -> tuple[str, str]:
    """
    Determina si una muestra está incluida, pendiente de revisión
    o fuera del alcance del proyecto.
    """
    fuente = inferir_fuente(video_path)

    if fuente == "MNNIT_h264":
        return "incluida", ""

    if fuente == "venta_sospechosa":
        return (
            "fuera_alcance",
            "Acción distinta al ocultamiento de mercancía por cliente",
        )

    return "pendiente_revision", ""


def extraer_metadata(video_path: Path) -> dict:
    """
    Extrae metadatos técnicos básicos de un archivo de video.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps and fps > 0:
        duracion_s = frames / fps
    else:
        duracion_s = None

    estado_muestra, motivo_exclusion = determinar_estado_muestra(video_path)

    return {
        "archivo": video_path.name,
        "clase": inferir_clase(video_path),
        "estado_muestra": estado_muestra,
        "motivo_exclusion": motivo_exclusion,
        "fps": round(fps, 3) if fps else None,
        "duracion_s": round(duracion_s, 3) if duracion_s else None,
        "ancho": ancho,
        "alto": alto,
        "frames": frames,
    }


def buscar_videos() -> list[Path]:
    """
    Recorre recursivamente /opt/workspace/datasets
    y devuelve todos los archivos de video encontrados.
    """
    videos = []

    for path in DATASETS_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return sorted(videos)


def main():
    videos = buscar_videos()

    print(f"Videos encontrados: {len(videos)}")

    registros = []
    contadores_fuente = {}

    for video_path in videos:
        fuente = inferir_fuente(video_path)

        contadores_fuente.setdefault(fuente, 0)
        contadores_fuente[fuente] += 1

        consecutivo = contadores_fuente[fuente]

        prefijo = (
            fuente.upper()
            .replace(" ", "_")
            .replace("-", "_")
        )

        video_id = f"{prefijo}_{consecutivo:03d}"

        try:
            metadata = extraer_metadata(video_path)

            registro = {
                "video_id": video_id,
                "fuente": fuente,
                **metadata,
            }

            registros.append(registro)

        except Exception as e:
            print(
                f"[ERROR] {video_path}: {e}"
            )

    df = pd.DataFrame(
        registros,
        columns=[
            "video_id",
            "fuente",
            "archivo",
            "clase",
            "estado_muestra",
            "motivo_exclusion",
            "fps",
            "duracion_s",
            "ancho",
            "alto",
            "frames",
        ],
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    print(f"\nCSV generado en:")
    print(OUTPUT_CSV)
    print(f"\nRegistros generados: {len(df)}")


if __name__ == "__main__":
    main()