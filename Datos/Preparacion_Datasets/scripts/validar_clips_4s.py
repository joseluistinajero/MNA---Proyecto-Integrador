import json
import subprocess
from pathlib import Path

CLIPS_DIR = Path(
    "/opt/workspace/datasets/procesados/clips_4s"
)

DURACION_OBJETIVO = 4.0
TOLERANCIA = 0.15


def analizar_clip(path):
    comando = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate",
        "-of", "json",
        str(path),
    ]

    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        check=True
    )

    return json.loads(resultado.stdout)


clips = sorted(CLIPS_DIR.glob("*.mp4"))

print("=" * 80)
print("VALIDACIÓN DE CLIPS GENERADOS")
print("=" * 80)

print(f"Clips encontrados: {len(clips)}")
print()

errores = []
duraciones_fuera = []
sin_video = []

for clip in clips:
    try:
        info = analizar_clip(clip)

        duracion = float(
            info.get("format", {}).get("duration", 0)
        )

        streams = info.get("streams", [])

        streams_video = [
            s for s in streams
            if s.get("codec_type") == "video"
        ]

        if not streams_video:
            sin_video.append(clip.name)

        if abs(duracion - DURACION_OBJETIVO) > TOLERANCIA:
            duraciones_fuera.append(
                (clip.name, duracion)
            )

    except Exception as e:
        errores.append(
            (clip.name, str(e))
        )


print("RESUMEN")
print("-" * 80)

print(f"Clips analizados:              {len(clips)}")
print(f"Errores de ffprobe:            {len(errores)}")
print(f"Clips sin stream de video:     {len(sin_video)}")
print(
    f"Duración fuera de tolerancia:  "
    f"{len(duraciones_fuera)}"
)

print()

if duraciones_fuera:
    print("CLIPS CON DURACIÓN FUERA DE TOLERANCIA")
    print("-" * 80)

    for nombre, duracion in duraciones_fuera:
        print(
            f"{nombre}: {duracion:.3f} s"
        )

    print()

if sin_video:
    print("CLIPS SIN STREAM DE VIDEO")
    print("-" * 80)

    for nombre in sin_video:
        print(nombre)

    print()

if errores:
    print("ERRORES DE FFPROBE")
    print("-" * 80)

    for nombre, error in errores:
        print(f"{nombre}: {error}")

    print()

if (
    len(clips) == 204
    and not errores
    and not sin_video
    and not duraciones_fuera
):
    print("=" * 80)
    print("RESULTADO: VALIDACIÓN SUPERADA")
    print("=" * 80)

else:
    print("=" * 80)
    print("RESULTADO: REVISIÓN NECESARIA")
    print("=" * 80)