import subprocess
from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_manifest_todos.csv"
)

CLIPS_DIR = Path(
    "/opt/workspace/datasets/procesados/clips_4s"
)

TOLERANCIA_4S = 0.15

FUENTES_RGB = {
    "CCTV_YOLO",
    "MNNIT",
    "SINTETICO",
    "DCSASS",
}


def limpiar(valor):
    return (
        str(valor)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def construir_nombre(row):
    return (
        f"{limpiar(row['dataset'])}__"
        f"{limpiar(row['video_id'])}__"
        f"{limpiar(row['clip_id'])}.mp4"
    )


def obtener_duracion(path):
    resultado = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(resultado.stdout.strip())


# ---------------------------------------------------------
# Seleccionar dataset consolidado
# ---------------------------------------------------------

df = pd.read_csv(MANIFEST_PATH)

df["dataset"] = (
    df["dataset"]
    .astype(str)
    .str.strip()
)

df["etiqueta"] = (
    df["etiqueta"]
    .astype(str)
    .str.strip()
)

consolidado = df[
    df["dataset"].isin(FUENTES_RGB)
].copy()

# DCSASS no_determinado no forma parte de clips_4s.
consolidado = consolidado[
    ~(
        consolidado["dataset"].eq("DCSASS")
        & consolidado["etiqueta"].eq("no_determinado")
    )
].copy()

consolidado["nombre_clip"] = consolidado.apply(
    construir_nombre,
    axis=1
)

consolidado["ruta_clip"] = consolidado[
    "nombre_clip"
].apply(
    lambda nombre: CLIPS_DIR / nombre
)


# ---------------------------------------------------------
# Medir duración real
# ---------------------------------------------------------

registros = []

for i, row in consolidado.iterrows():

    path = row["ruta_clip"]

    if not path.is_file():
        print(f"ERROR: no existe {path}")
        raise SystemExit(1)

    duracion = obtener_duracion(path)

    registros.append({
        "dataset": row["dataset"],
        "video_id": row["video_id"],
        "clip_id": row["clip_id"],
        "etiqueta": row["etiqueta"],
        "nombre_clip": row["nombre_clip"],
        "duracion_real_s": duracion,
    })


resultado = pd.DataFrame(registros)

resultado["cumple_4s"] = (
    (resultado["duracion_real_s"] - 4.0)
    .abs()
    .le(TOLERANCIA_4S)
)

resultado["diferencia_vs_4s"] = (
    resultado["duracion_real_s"] - 4.0
)


# ---------------------------------------------------------
# Resumen general
# ---------------------------------------------------------

print("=" * 80)
print("CARACTERIZACIÓN TEMPORAL DEL DATASET CONSOLIDADO")
print("=" * 80)

print(f"Clips analizados: {len(resultado)}")
print(
    f"Duración mínima: "
    f"{resultado['duracion_real_s'].min():.3f} s"
)
print(
    f"Duración máxima: "
    f"{resultado['duracion_real_s'].max():.3f} s"
)
print(
    f"Duración media:  "
    f"{resultado['duracion_real_s'].mean():.3f} s"
)
print(
    f"Mediana:         "
    f"{resultado['duracion_real_s'].median():.3f} s"
)

print(
    f"Clips ~4 s:      "
    f"{resultado['cumple_4s'].sum()}"
)

print(
    f"Clips != ~4 s:   "
    f"{(~resultado['cumple_4s']).sum()}"
)


# ---------------------------------------------------------
# Por dataset
# ---------------------------------------------------------

print()
print("RESUMEN POR DATASET")
print("-" * 80)

resumen_dataset = (
    resultado
    .groupby("dataset")
    .agg(
        clips=("nombre_clip", "count"),
        duracion_min=("duracion_real_s", "min"),
        duracion_max=("duracion_real_s", "max"),
        duracion_media=("duracion_real_s", "mean"),
        mediana=("duracion_real_s", "median"),
        clips_4s=("cumple_4s", "sum"),
    )
)

resumen_dataset["fuera_4s"] = (
    resumen_dataset["clips"]
    - resumen_dataset["clips_4s"]
)

print(
    resumen_dataset
    .round(3)
    .to_string()
)


# ---------------------------------------------------------
# Por etiqueta
# ---------------------------------------------------------

print()
print("RESUMEN POR ETIQUETA")
print("-" * 80)

resumen_etiqueta = (
    resultado
    .groupby("etiqueta")
    .agg(
        clips=("nombre_clip", "count"),
        duracion_min=("duracion_real_s", "min"),
        duracion_max=("duracion_real_s", "max"),
        duracion_media=("duracion_real_s", "mean"),
        clips_4s=("cumple_4s", "sum"),
    )
)

resumen_etiqueta["fuera_4s"] = (
    resumen_etiqueta["clips"]
    - resumen_etiqueta["clips_4s"]
)

print(
    resumen_etiqueta
    .round(3)
    .to_string()
)


# ---------------------------------------------------------
# Distribución de duraciones redondeadas
# ---------------------------------------------------------

print()
print("DISTRIBUCIÓN DE DURACIONES")
print("-" * 80)

resultado["duracion_redondeada"] = (
    resultado["duracion_real_s"]
    .round(2)
)

distribucion = (
    resultado["duracion_redondeada"]
    .value_counts()
    .sort_index()
)

for duracion, cantidad in distribucion.items():
    print(
        f"{duracion:7.2f} s : "
        f"{cantidad:4d} clips"
    )


# ---------------------------------------------------------
# Guardar mediciones para análisis posterior
# ---------------------------------------------------------

SALIDA = (
    MANIFEST_PATH.parent
    / "duraciones_dataset_consolidado.csv"
)

resultado.to_csv(
    SALIDA,
    index=False
)

print()
print("=" * 80)
print("RESULTADO")
print("=" * 80)
print(f"Archivo generado: {SALIDA}")
print(f"Filas guardadas:  {len(resultado)}")

if len(resultado) == 987:
    print(
        "RESULTADO: CARACTERIZACIÓN COMPLETADA"
    )
else:
    print(
        "RESULTADO: REVISIÓN NECESARIA"
    )