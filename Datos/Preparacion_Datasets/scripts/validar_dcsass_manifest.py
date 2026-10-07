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


# ---------------------------------------------------------
# Leer manifiesto
# ---------------------------------------------------------

df = pd.read_csv(MANIFEST_PATH)

dcsass = df[
    df["dataset"].astype(str).str.strip() == "DCSASS"
].copy()


print("=" * 80)
print("DCSASS EN EL MANIFIESTO")
print("=" * 80)

print(f"Registros DCSASS: {len(dcsass)}")
print()

print("COLUMNAS")
print("-" * 80)

for columna in dcsass.columns:
    print(columna)


print()
print("PRIMEROS 15 REGISTROS")
print("-" * 80)

columnas_mostrar = [
    c for c in [
        "dataset",
        "video_id",
        "video_origen",
        "clip_id",
        "inicio_s",
        "fin_s",
        "etiqueta"
    ]
    if c in dcsass.columns
]

print(
    dcsass[columnas_mostrar]
    .head(15)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Inventario físico DCSASS
# ---------------------------------------------------------

archivos = sorted(
    p for p in DCSASS_DIR.rglob("*.mp4")
    if p.is_file()
)

nombres_archivo = {
    p.name
    for p in archivos
}


print()
print("=" * 80)
print("INVENTARIO FÍSICO")
print("=" * 80)

print(f"Archivos MP4 disponibles: {len(archivos)}")


# ---------------------------------------------------------
# Probar correspondencia directa con columnas del manifiesto
# ---------------------------------------------------------

print()
print("=" * 80)
print("PRUEBAS DE CORRESPONDENCIA")
print("=" * 80)

for columna in ["video_origen", "video_id", "clip_id"]:

    if columna not in dcsass.columns:
        continue

    valores = (
        dcsass[columna]
        .dropna()
        .astype(str)
        .str.strip()
    )

    coincidencias = valores.isin(nombres_archivo).sum()

    print(
        f"{columna:15s}: "
        f"{coincidencias} coincidencias directas "
        f"de {len(valores)}"
    )


# ---------------------------------------------------------
# Mostrar archivos físicos de ejemplo
# ---------------------------------------------------------

print()
print("PRIMEROS 15 ARCHIVOS FÍSICOS")
print("-" * 80)

for path in archivos[:15]:
    print(path.relative_to(DCSASS_DIR))