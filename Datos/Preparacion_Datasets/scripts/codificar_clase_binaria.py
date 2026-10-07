import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/"
    "dataset_consolidado_features.csv"
)

MAPEO_CLASE = {
    "sin_ocultamiento": 0,
    "ocultamiento": 1,
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Agrega la codificación numérica de la clase "
            "binaria al dataset consolidado."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Guarda la nueva columna en el CSV. "
            "Sin esta opción solo realiza validación."
        ),
    )

    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"ERROR: no existe {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)

    if "etiqueta" not in df.columns:
        print("ERROR: falta la columna etiqueta.")
        sys.exit(1)

    df["etiqueta"] = (
        df["etiqueta"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Codificación
    # -----------------------------------------------------

    df["clase_binaria"] = (
        df["etiqueta"]
        .map(MAPEO_CLASE)
        .astype("Int64")
    )

    # -----------------------------------------------------
    # Validaciones
    # -----------------------------------------------------

    conteo_0 = int(
        (df["clase_binaria"] == 0).sum()
    )

    conteo_1 = int(
        (df["clase_binaria"] == 1).sum()
    )

    conteo_na = int(
        df["clase_binaria"].isna().sum()
    )

    etiquetas_sin_codificar = sorted(
        df.loc[
            df["clase_binaria"].isna(),
            "etiqueta",
        ]
        .unique()
        .tolist()
    )

    print("=" * 80)
    print("CODIFICACIÓN DE CLASE BINARIA")
    print("=" * 80)

    print(f"Registros totales:           {len(df)}")
    print(f"Clase 0 - sin_ocultamiento:  {conteo_0}")
    print(f"Clase 1 - ocultamiento:      {conteo_1}")
    print(f"Sin clase binaria:           {conteo_na}")

    print()
    print("ETIQUETAS SIN CODIFICACIÓN BINARIA")
    print("-" * 80)

    for etiqueta in etiquetas_sin_codificar:
        cantidad = int(
            (
                (df["etiqueta"] == etiqueta)
                & df["clase_binaria"].isna()
            ).sum()
        )

        print(
            f"{etiqueta}: {cantidad}"
        )

    print()
    print("PRIMEROS 10 REGISTROS")
    print("-" * 80)

    print(
        df[
            [
                "dataset",
                "clip_id",
                "etiqueta",
                "clase_binaria",
                "elegible_modelo_binario",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Validación esperada
    # -----------------------------------------------------

    correcto = (
        len(df) == 987
        and conteo_0 == 795
        and conteo_1 == 190
        and conteo_na == 2
        and etiquetas_sin_codificar == ["ambiguo"]
    )

    print()

    if not correcto:
        print("RESULTADO: REVISIÓN NECESARIA")
        sys.exit(1)

    if not args.execute:
        print("=" * 80)
        print("DRY-RUN COMPLETADO")
        print("No se modificó dataset_consolidado_features.csv")
        print("=" * 80)
        return

    # -----------------------------------------------------
    # Respaldo
    # -----------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = DATASET_PATH.with_name(
        f"{DATASET_PATH.stem}_backup_"
        f"{timestamp}{DATASET_PATH.suffix}"
    )

    shutil.copy2(
        DATASET_PATH,
        backup,
    )

    # -----------------------------------------------------
    # Guardar
    # -----------------------------------------------------

    df.to_csv(
        DATASET_PATH,
        index=False,
    )

    print("=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(f"Respaldo: {backup}")
    print(f"Archivo actualizado: {DATASET_PATH}")
    print("RESULTADO: CODIFICACIÓN VALIDADA")


if __name__ == "__main__":
    main()