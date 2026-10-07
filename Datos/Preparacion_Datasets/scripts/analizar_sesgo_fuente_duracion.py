from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/"
    "duraciones_dataset_consolidado.csv"
)

ETIQUETAS_BINARIAS = {
    "ocultamiento",
    "sin_ocultamiento",
}


df = pd.read_csv(INPUT_PATH)

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

# Conservamos el consolidado original intacto.
# Para este análisis utilizamos únicamente las dos clases
# del problema binario.
binario = df[
    df["etiqueta"].isin(ETIQUETAS_BINARIAS)
].copy()

binario["duracion_redondeada"] = (
    binario["duracion_real_s"]
    .round()
    .astype(int)
)


print("=" * 80)
print("ANÁLISIS DE POSIBLE SESGO: FUENTE, DURACIÓN Y ETIQUETA")
print("=" * 80)

print(f"Clips consolidados:       {len(df)}")
print(f"Clips análisis binario:   {len(binario)}")
print(f"Clips fuera del binario:  {len(df) - len(binario)}")


# =========================================================
# 1. Etiqueta por dataset
# =========================================================

print()
print("1. DISTRIBUCIÓN DE ETIQUETAS POR DATASET")
print("-" * 80)

tabla_dataset = pd.crosstab(
    binario["dataset"],
    binario["etiqueta"],
    margins=True,
)

print(tabla_dataset.to_string())


print()
print("PORCENTAJE DE OCULTAMIENTO POR DATASET")
print("-" * 80)

for dataset, grupo in (
    binario.groupby("dataset")
):
    total = len(grupo)

    ocultamiento = (
        grupo["etiqueta"]
        .eq("ocultamiento")
        .sum()
    )

    porcentaje = (
        ocultamiento / total * 100
        if total
        else 0
    )

    print(
        f"{dataset:15s} "
        f"n={total:4d} "
        f"ocultamiento={ocultamiento:4d} "
        f"({porcentaje:6.2f}%)"
    )


# =========================================================
# 2. Etiqueta por duración
# =========================================================

print()
print()
print("2. DISTRIBUCIÓN DE ETIQUETAS POR DURACIÓN")
print("-" * 80)

tabla_duracion = pd.crosstab(
    binario["duracion_redondeada"],
    binario["etiqueta"],
    margins=True,
)

print(tabla_duracion.to_string())


print()
print("PORCENTAJE DE OCULTAMIENTO POR DURACIÓN")
print("-" * 80)

for duracion, grupo in (
    binario.groupby("duracion_redondeada")
):
    total = len(grupo)

    ocultamiento = (
        grupo["etiqueta"]
        .eq("ocultamiento")
        .sum()
    )

    porcentaje = (
        ocultamiento / total * 100
        if total
        else 0
    )

    print(
        f"{duracion:2d} s "
        f"n={total:4d} "
        f"ocultamiento={ocultamiento:4d} "
        f"({porcentaje:6.2f}%)"
    )


# =========================================================
# 3. DCSASS aislado
# =========================================================

dcsass = binario[
    binario["dataset"].eq("DCSASS")
].copy()

print()
print()
print("3. DCSASS: ETIQUETA POR DURACIÓN")
print("-" * 80)

tabla_dcsass = pd.crosstab(
    dcsass["duracion_redondeada"],
    dcsass["etiqueta"],
    margins=True,
)

print(tabla_dcsass.to_string())


print()
print("DCSASS: PORCENTAJE DE OCULTAMIENTO POR DURACIÓN")
print("-" * 80)

for duracion, grupo in (
    dcsass.groupby("duracion_redondeada")
):
    total = len(grupo)

    ocultamiento = (
        grupo["etiqueta"]
        .eq("ocultamiento")
        .sum()
    )

    porcentaje = (
        ocultamiento / total * 100
        if total
        else 0
    )

    print(
        f"{duracion:2d} s "
        f"n={total:4d} "
        f"ocultamiento={ocultamiento:4d} "
        f"({porcentaje:6.2f}%)"
    )


# =========================================================
# 4. Duración promedio por dataset y etiqueta
# =========================================================

print()
print()
print("4. DURACIÓN MEDIA POR DATASET Y ETIQUETA")
print("-" * 80)

resumen = (
    binario
    .groupby(
        ["dataset", "etiqueta"]
    )
    .agg(
        clips=("clip_id", "count"),
        duracion_media=(
            "duracion_real_s",
            "mean"
        ),
        duracion_mediana=(
            "duracion_real_s",
            "median"
        ),
        duracion_min=(
            "duracion_real_s",
            "min"
        ),
        duracion_max=(
            "duracion_real_s",
            "max"
        ),
    )
)

print(
    resumen
    .round(3)
    .to_string()
)


print()
print("=" * 80)
print("RESULTADO: ANÁLISIS COMPLETADO")
print("=" * 80)