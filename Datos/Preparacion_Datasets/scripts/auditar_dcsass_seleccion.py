from pathlib import Path
import pandas as pd


MANIFEST_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_manifest_todos.csv"
)

DCSASS_DIR = Path(
    "/opt/workspace/datasets/"
    "Anomaly Detection Dataset UCF/"
    "Shoplifting_Clips"
)


df = pd.read_csv(MANIFEST_PATH)

dcsass = df[
    df["dataset"].astype(str).str.strip() == "DCSASS"
].copy()

dcsass["etiqueta"] = (
    dcsass["etiqueta"]
    .astype(str)
    .str.strip()
)

print("=" * 80)
print("AUDITORÍA DCSASS PARA DATASET OPERATIVO")
print("=" * 80)

print(f"Registros DCSASS totales: {len(dcsass)}")


# ---------------------------------------------------------
# Distribución de etiquetas
# ---------------------------------------------------------

print()
print("DISTRIBUCIÓN DE ETIQUETAS")
print("-" * 80)

conteo_etiquetas = (
    dcsass["etiqueta"]
    .value_counts(dropna=False)
)

print(conteo_etiquetas.to_string())


# ---------------------------------------------------------
# Selección para modelo binario
# ---------------------------------------------------------

ETIQUETAS_MODELO = {
    "ocultamiento",
    "sin_ocultamiento"
}

seleccionados = dcsass[
    dcsass["etiqueta"].isin(ETIQUETAS_MODELO)
].copy()

excluidos = dcsass[
    ~dcsass["etiqueta"].isin(ETIQUETAS_MODELO)
].copy()

print()
print("SELECCIÓN PARA clips_4s")
print("-" * 80)

print(f"Clips elegibles:             {len(seleccionados)}")
print(f"Clips excluidos:             {len(excluidos)}")

print()
print("ETIQUETAS EXCLUIDAS")
print("-" * 80)

if excluidos.empty:
    print("Ninguna")
else:
    print(
        excluidos["etiqueta"]
        .value_counts(dropna=False)
        .to_string()
    )


# ---------------------------------------------------------
# Duplicados
# ---------------------------------------------------------

duplicados = dcsass[
    dcsass["video_origen"].duplicated(keep=False)
]

print()
print("DUPLICADOS")
print("-" * 80)

print(
    f"Registros con video_origen duplicado: "
    f"{len(duplicados)}"
)


# ---------------------------------------------------------
# Convención temporal actual
# ---------------------------------------------------------

inicio = pd.to_numeric(
    dcsass["inicio_s"],
    errors="coerce"
)

fin = pd.to_numeric(
    dcsass["fin_s"],
    errors="coerce"
)

patron_0_3 = (
    (inicio == 0)
    & (fin == 3)
)

print()
print("CONVENCIÓN TEMPORAL ACTUAL")
print("-" * 80)

print(
    f"Registros con inicio_s=0 y fin_s=3: "
    f"{patron_0_3.sum()}"
)

print(
    f"Registros con otro patrón temporal: "
    f"{(~patron_0_3).sum()}"
)


# ---------------------------------------------------------
# Existencia física de seleccionados
# ---------------------------------------------------------

archivos_fisicos = {
    p.name
    for p in DCSASS_DIR.rglob("*.mp4")
    if p.is_file()
}

seleccionados["archivo_existe"] = (
    seleccionados["video_origen"]
    .astype(str)
    .str.strip()
    .isin(archivos_fisicos)
)

faltantes = seleccionados[
    ~seleccionados["archivo_existe"]
]

print()
print("VALIDACIÓN FÍSICA")
print("-" * 80)

print(
    f"Clips elegibles encontrados: "
    f"{seleccionados['archivo_existe'].sum()}"
)

print(
    f"Clips elegibles faltantes:   "
    f"{len(faltantes)}"
)


# ---------------------------------------------------------
# Resultado
# ---------------------------------------------------------

print()
print("=" * 80)

if (
    len(duplicados) == 0
    and len(faltantes) == 0
):
    print("RESULTADO: SELECCIÓN DCSASS VALIDADA")
else:
    print("RESULTADO: REVISIÓN NECESARIA")

print("=" * 80)
