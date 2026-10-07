from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_manifest_todos.csv"
)

DURACIONES_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/duraciones_dataset_consolidado.csv"
)

SALIDA_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_consolidado_features.csv"
)

ETIQUETAS_BINARIAS = {
    "ocultamiento",
    "sin_ocultamiento",
}

DURACION_OBJETIVO = 4.0


# ---------------------------------------------------------
# Cargar archivos
# ---------------------------------------------------------

manifest = pd.read_csv(MANIFEST_PATH)
duraciones = pd.read_csv(DURACIONES_PATH)

for df in [manifest, duraciones]:
    df["dataset"] = (
        df["dataset"]
        .astype(str)
        .str.strip()
    )

    df["clip_id"] = (
        df["clip_id"]
        .astype(str)
        .str.strip()
    )


# ---------------------------------------------------------
# Seleccionar únicamente los 987 clips físicos
# ---------------------------------------------------------

claves_consolidado = (
    duraciones[
        [
            "dataset",
            "clip_id",
        ]
    ]
    .drop_duplicates()
)

consolidado = manifest.merge(
    claves_consolidado,
    on=[
        "dataset",
        "clip_id",
    ],
    how="inner",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Incorporar duración medida
# ---------------------------------------------------------

variables_duracion = duraciones[
    [
        "dataset",
        "clip_id",
        "nombre_clip",
        "duracion_real_s",
        "cumple_4s",
        "diferencia_vs_4s",
    ]
].copy()

consolidado = consolidado.merge(
    variables_duracion,
    on=[
        "dataset",
        "clip_id",
    ],
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Nuevas características
# ---------------------------------------------------------

consolidado["duracion_objetivo_s"] = (
    DURACION_OBJETIVO
)

consolidado[
    "requiere_normalizacion_temporal"
] = ~consolidado["cumple_4s"].astype(bool)


consolidado["etiqueta"] = (
    consolidado["etiqueta"]
    .astype(str)
    .str.strip()
)

consolidado["elegible_modelo_binario"] = (
    consolidado["etiqueta"]
    .isin(ETIQUETAS_BINARIAS)
)


def motivo_no_elegible(row):
    if row["elegible_modelo_binario"]:
        return ""

    return f"etiqueta_{row['etiqueta']}"


consolidado["motivo_no_elegible"] = (
    consolidado.apply(
        motivo_no_elegible,
        axis=1,
    )
)


# ---------------------------------------------------------
# Reordenar columnas
# ---------------------------------------------------------

columnas_base = [
    "dataset",
    "video_id",
    "video_origen",
    "clip_id",
    "inicio_s",
    "fin_s",
    "etiqueta",
    "persona_id",
    "evento_id",
    "nota",
]

columnas_features = [
    "nombre_clip",
    "duracion_real_s",
    "duracion_objetivo_s",
    "diferencia_vs_4s",
    "cumple_4s",
    "requiere_normalizacion_temporal",
    "elegible_modelo_binario",
    "motivo_no_elegible",
]

columnas_finales = [
    c
    for c in (
        columnas_base
        + columnas_features
    )
    if c in consolidado.columns
]

consolidado = consolidado[
    columnas_finales
].copy()


# ---------------------------------------------------------
# Validaciones
# ---------------------------------------------------------

print("=" * 80)
print("CONSTRUCCIÓN DEL DATASET CONSOLIDADO ENRIQUECIDO")
print("=" * 80)

print(
    f"Registros consolidados:              "
    f"{len(consolidado)}"
)

print(
    f"Duraciones faltantes:                "
    f"{consolidado['duracion_real_s'].isna().sum()}"
)

print(
    f"Clips que cumplen ~4 s:              "
    f"{consolidado['cumple_4s'].sum()}"
)

print(
    f"Requieren normalización temporal:    "
    f"{consolidado['requiere_normalizacion_temporal'].sum()}"
)

print(
    f"Elegibles para modelo binario:       "
    f"{consolidado['elegible_modelo_binario'].sum()}"
)

print(
    f"No elegibles para modelo binario:    "
    f"{(~consolidado['elegible_modelo_binario']).sum()}"
)


print()
print("NO ELEGIBLES")
print("-" * 80)

no_elegibles = consolidado[
    ~consolidado["elegible_modelo_binario"]
]

if no_elegibles.empty:
    print("Ninguno")
else:
    print(
        no_elegibles[
            [
                "dataset",
                "clip_id",
                "etiqueta",
                "motivo_no_elegible",
            ]
        ].to_string(index=False)
    )


print()
print("RESUMEN POR DATASET")
print("-" * 80)

resumen = (
    consolidado
    .groupby("dataset")
    .agg(
        clips=("clip_id", "count"),
        cumple_4s=("cumple_4s", "sum"),
        requiere_normalizacion=(
            "requiere_normalizacion_temporal",
            "sum",
        ),
        elegibles_binario=(
            "elegible_modelo_binario",
            "sum",
        ),
    )
)

print(resumen.to_string())


# ---------------------------------------------------------
# Guardar
# ---------------------------------------------------------

consolidado.to_csv(
    SALIDA_PATH,
    index=False,
)


print()
print("=" * 80)
print("RESULTADO")
print("=" * 80)

print(f"Archivo generado: {SALIDA_PATH}")
print(f"Filas guardadas:  {len(consolidado)}")


correcto = (
    len(consolidado) == 987
    and consolidado[
        "duracion_real_s"
    ].isna().sum() == 0
    and consolidado[
        "requiere_normalizacion_temporal"
    ].sum() == 762
    and consolidado[
        "elegible_modelo_binario"
    ].sum() == 985
)


if correcto:
    print(
        "RESULTADO: DATASET CONSOLIDADO "
        "ENRIQUECIDO VALIDADO"
    )
else:
    print(
        "RESULTADO: REVISIÓN NECESARIA"
    )
    raise SystemExit(1)