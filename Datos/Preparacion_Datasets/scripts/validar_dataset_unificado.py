import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_manifest_todos.csv"
)

CLIPS_DIR = Path(
    "/opt/workspace/datasets/procesados/clips_4s"
)

ETIQUETAS_MODELO = {
    "ocultamiento",
    "sin_ocultamiento",
}

DURACION_OBJETIVO = 4.0
TOLERANCIA = 0.15
TOTAL_ESPERADO = 987


def limpiar(valor):
    return (
        str(valor)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def nombre_salida(row):
    return (
        f"{limpiar(row['dataset'])}__"
        f"{limpiar(row['video_id'])}__"
        f"{limpiar(row['clip_id'])}.mp4"
    )


def obtener_duracion(path):
    comando = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(resultado.stdout.strip())


def validar_decodificacion(path):
    comando = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-map", "0:v:0",
        "-f", "null",
        "-",
    ]

    resultado = subprocess.run(
        comando,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    return (
        resultado.returncode == 0,
        resultado.stderr.strip(),
    )


# =========================================================
# 1. Construir conjunto esperado desde el manifiesto
# =========================================================

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

# RetailS y otras fuentes que no pertenecen al dataset RGB
# quedan fuera porque solo consideramos las fuentes que
# realmente fueron materializadas en clips_4s.
fuentes_rgb = {
    "CCTV_YOLO",
    "MNNIT",
    "SINTETICO",
    "DCSASS",
}

esperados = df[
    df["dataset"].isin(fuentes_rgb)
].copy()

# Para DCSASS se excluye no_determinado.
mask_dcsass_no_modelo = (
    esperados["dataset"].eq("DCSASS")
    & ~esperados["etiqueta"].isin(ETIQUETAS_MODELO)
)

esperados = esperados[
    ~mask_dcsass_no_modelo
].copy()

esperados["nombre_clip"] = esperados.apply(
    nombre_salida,
    axis=1,
)

nombres_esperados = set(
    esperados["nombre_clip"]
)


# =========================================================
# 2. Inventario físico
# =========================================================

archivos = sorted(
    p for p in CLIPS_DIR.glob("*.mp4")
    if p.is_file()
)

nombres_reales = {
    p.name
    for p in archivos
}

faltantes = sorted(
    nombres_esperados - nombres_reales
)

extras = sorted(
    nombres_reales - nombres_esperados
)

duplicados_manifest = esperados[
    esperados["nombre_clip"].duplicated(
        keep=False
    )
]


print("=" * 80)
print("VALIDACIÓN DEL DATASET UNIFICADO clips_4s")
print("=" * 80)

print(f"Clips esperados por manifiesto: {len(esperados)}")
print(f"Archivos físicos encontrados:   {len(archivos)}")
print(f"Archivos faltantes:             {len(faltantes)}")
print(f"Archivos extras:                {len(extras)}")
print(
    f"Nombres duplicados manifiesto:  "
    f"{len(duplicados_manifest)}"
)


# =========================================================
# 3. Conteo por dataset
# =========================================================

conteo_manifest = (
    esperados["dataset"]
    .value_counts()
    .sort_index()
)

conteo_fisico = Counter()

for archivo in archivos:
    dataset = archivo.name.split("__", 1)[0]
    conteo_fisico[dataset] += 1


print()
print("CONTEO POR DATASET")
print("-" * 80)

datasets = sorted(
    set(conteo_manifest.index)
    | set(conteo_fisico.keys())
)

for dataset in datasets:
    esperado = int(
        conteo_manifest.get(dataset, 0)
    )

    real = conteo_fisico.get(dataset, 0)

    print(
        f"{dataset:15s} "
        f"esperados={esperado:4d} "
        f"físicos={real:4d}"
    )


# =========================================================
# 4. Distribución de clases
# =========================================================

print()
print("DISTRIBUCIÓN DE ETIQUETAS")
print("-" * 80)

print(
    esperados["etiqueta"]
    .value_counts()
    .to_string()
)


# =========================================================
# 5. Duración
# =========================================================

duraciones_fuera = []
errores_probe = []

print()
print("VALIDANDO DURACIONES...")

for i, archivo in enumerate(
    archivos,
    start=1,
):
    try:
        duracion = obtener_duracion(
            archivo
        )

        if abs(
            duracion - DURACION_OBJETIVO
        ) > TOLERANCIA:
            duraciones_fuera.append(
                (
                    archivo.name,
                    duracion,
                )
            )

    except Exception as e:
        errores_probe.append(
            (
                archivo.name,
                str(e),
            )
        )

    if i % 100 == 0:
        print(
            f"  {i}/{len(archivos)}"
        )


# =========================================================
# 6. Decodificación completa
# =========================================================

errores_decode = []

print()
print("VALIDANDO DECODIFICACIÓN...")

for i, archivo in enumerate(
    archivos,
    start=1,
):
    correcto, error = (
        validar_decodificacion(
            archivo
        )
    )

    if not correcto:
        errores_decode.append(
            (
                archivo.name,
                error,
            )
        )

    if i % 100 == 0:
        print(
            f"  {i}/{len(archivos)}"
        )


# =========================================================
# 7. Resumen final
# =========================================================

print()
print("=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print(
    f"Clips físicos:                   "
    f"{len(archivos)}"
)

print(
    f"Faltantes contra manifiesto:     "
    f"{len(faltantes)}"
)

print(
    f"Extras contra manifiesto:        "
    f"{len(extras)}"
)

print(
    f"Duración fuera de tolerancia:    "
    f"{len(duraciones_fuera)}"
)

print(
    f"Errores de ffprobe:              "
    f"{len(errores_probe)}"
)

print(
    f"Errores de decodificación:       "
    f"{len(errores_decode)}"
)


if duraciones_fuera:
    print()
    print("DURACIONES FUERA DE TOLERANCIA")
    print("-" * 80)

    for nombre, duracion in (
        duraciones_fuera[:20]
    ):
        print(
            f"{nombre}: "
            f"{duracion:.3f} s"
        )


if faltantes:
    print()
    print("PRIMEROS ARCHIVOS FALTANTES")
    print("-" * 80)

    for nombre in faltantes[:20]:
        print(nombre)


if extras:
    print()
    print("PRIMEROS ARCHIVOS EXTRAS")
    print("-" * 80)

    for nombre in extras[:20]:
        print(nombre)


correcto = (
    len(esperados) == TOTAL_ESPERADO
    and len(archivos) == TOTAL_ESPERADO
    and len(faltantes) == 0
    and len(extras) == 0
    and len(duplicados_manifest) == 0
    and len(duraciones_fuera) == 0
    and len(errores_probe) == 0
    and len(errores_decode) == 0
)

print()
print("=" * 80)

if correcto:
    print(
        "RESULTADO: DATASET UNIFICADO VALIDADO"
    )
else:
    print(
        "RESULTADO: REVISIÓN NECESARIA"
    )
    sys.exit(1)

print("=" * 80)