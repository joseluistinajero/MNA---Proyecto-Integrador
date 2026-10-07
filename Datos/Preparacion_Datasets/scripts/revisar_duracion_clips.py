import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

MANIFEST_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_manifest_todos.csv"
)

METADATA_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/videos_metadata.csv"
)

OUTPUT_PATH = MANIFEST_PATH.parent / "clips_fuera_duracion.csv"


# ---------------------------------------------------------
# Cargar archivos
# ---------------------------------------------------------

manifest = pd.read_csv(MANIFEST_PATH)
metadata = pd.read_csv(METADATA_PATH)


# ---------------------------------------------------------
# Normalizar nombres de archivo para poder relacionarlos
#
# dataset_manifest_todos.csv:
#     video_origen
#
# videos_metadata.csv:
#     archivo
# ---------------------------------------------------------

manifest["archivo_key"] = (
    manifest["video_origen"]
    .astype(str)
    .str.strip()
    .str.lower()
)

metadata["archivo_key"] = (
    metadata["archivo"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ---------------------------------------------------------
# Agregar metadata del video al manifiesto
# ---------------------------------------------------------

df = manifest.merge(
    metadata[
        [
            "archivo_key",
            "video_id",
            "fuente",
            "archivo",
            "duracion_s",
            "fps",
            "frames"
        ]
    ],
    on="archivo_key",
    how="left",
    suffixes=("_manifest", "_metadata")
)


# ---------------------------------------------------------
# Calcular duración del clip
#
# Convención:
#
# [inicio_s, fin_s)
#
# Ejemplo:
#
# inicio_s = 2
# fin_s    = 6
#
# El clip cubre desde 2.000 s hasta justo antes de 6.000 s
#
# duración = 6 - 2 = 4 s
# ---------------------------------------------------------

df["duracion_clip_s"] = (
    df["fin_s"] - df["inicio_s"]
)


# ---------------------------------------------------------
# Detectar clips inválidos por duración
#
# Un clip excede el video únicamente si:
#
# fin_s > duracion_s
#
# Ejemplo válido:
#     fin_s = 6.000
#     video = 6.042
#
# Ejemplo inválido:
#     fin_s = 7.000
#     video = 6.042
# ---------------------------------------------------------

fuera_duracion = df[
    df["duracion_s"].notna()
    & (df["fin_s"] > df["duracion_s"])
].copy()


# ---------------------------------------------------------
# Calcular cuánto excede
# ---------------------------------------------------------

fuera_duracion["exceso_s"] = (
    fuera_duracion["fin_s"]
    - fuera_duracion["duracion_s"]
)


# ---------------------------------------------------------
# Seleccionar columnas útiles
# ---------------------------------------------------------

columnas_salida = [
    "dataset",
    "video_id_manifest",
    "video_origen",
    "clip_id",
    "inicio_s",
    "fin_s",
    "duracion_clip_s",
    "duracion_s",
    "exceso_s",
    "etiqueta",
    "fuente",
]

fuera_duracion = fuera_duracion[columnas_salida]


# ---------------------------------------------------------
# Mostrar resumen
# ---------------------------------------------------------

print("=" * 80)
print("VERIFICACIÓN DE DURACIÓN DE CLIPS")
print("Convención temporal: [inicio_s, fin_s)")
print("=" * 80)

print(f"Clips en manifiesto: {len(manifest)}")
print(f"Videos en metadata:  {len(metadata)}")

print(
    f"Clips con metadata encontrada: "
    f"{df['duracion_s'].notna().sum()}"
)

print(
    f"Clips SIN metadata encontrada: "
    f"{df['duracion_s'].isna().sum()}"
)

print(
    f"Clips que exceden la duración del video: "
    f"{len(fuera_duracion)}"
)

print()


# ---------------------------------------------------------
# Mostrar y guardar clips fuera de duración
# ---------------------------------------------------------

if fuera_duracion.empty:
    print("No se encontraron clips fuera de la duración del video.")

else:
    print(fuera_duracion.to_string(index=False))

    fuera_duracion.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Resultado guardado en:")
    print(OUTPUT_PATH)


# ---------------------------------------------------------
# Mostrar clips sin metadata asociada
# ---------------------------------------------------------

sin_metadata = df[
    df["duracion_s"].isna()
][
    [
        "dataset",
        "video_id_manifest",
        "video_origen",
        "clip_id",
        "inicio_s",
        "fin_s",
    ]
].copy()

if not sin_metadata.empty:
    print()
    print("=" * 80)
    print("CLIPS SIN COINCIDENCIA EN videos_metadata.csv")
    print("=" * 80)

    print(sin_metadata.to_string(index=False))


# ---------------------------------------------------------
# Validación adicional:
# detectar clips con duración <= 0
# ---------------------------------------------------------

clips_invalidos = df[
    df["duracion_clip_s"] <= 0
][
    [
        "dataset",
        "video_id_manifest",
        "video_origen",
        "clip_id",
        "inicio_s",
        "fin_s",
        "duracion_clip_s",
    ]
].copy()

if not clips_invalidos.empty:
    print()
    print("=" * 80)
    print("CLIPS CON DURACIÓN INVÁLIDA")
    print("=" * 80)

    print(clips_invalidos.to_string(index=False))