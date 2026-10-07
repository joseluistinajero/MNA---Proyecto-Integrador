import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_manifest_todos.csv"
)

DCSASS_DIR = Path(
    "/opt/workspace/datasets/Anomaly Detection Dataset UCF"
)

OUTPUT_DIR = Path(
    "/opt/workspace/datasets/procesados/clips_4s"
)


def construir_nombre_clip(row):
    dataset = str(row["dataset"]).strip()
    video_id = str(row["video_id"]).strip()
    clip_id = str(row["clip_id"]).strip()

    def limpiar(valor):
        return (
            valor
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

    return (
        f"{limpiar(dataset)}__"
        f"{limpiar(video_id)}__"
        f"{limpiar(clip_id)}.mp4"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Incorpora al dataset unificado los clips DCSASS "
            "ya existentes."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copia realmente los archivos. Sin esta opción solo hace dry-run."
    )

    args = parser.parse_args()

    modo = "EJECUCIÓN" if args.execute else "DRY-RUN"

    print("=" * 80)
    print(f"INCORPORACIÓN DE DCSASS - {modo}")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        print("ERROR: no existe el manifiesto.")
        sys.exit(1)

    if not DCSASS_DIR.exists():
        print("ERROR: no existe la carpeta fuente DCSASS.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)

    # -----------------------------------------------------
    # Seleccionar exclusivamente DCSASS
    # -----------------------------------------------------

    df = df[
        df["dataset"].astype(str).str.strip() == "DCSASS"
    ].copy()

    print(f"Registros DCSASS en el manifiesto: {len(df)}")

    if df.empty:
        print("ERROR: no hay registros DCSASS en el manifiesto.")
        sys.exit(1)

    # -----------------------------------------------------
    # Construir rutas
    # -----------------------------------------------------

    df["ruta_origen"] = df["video_origen"].apply(
        lambda nombre: DCSASS_DIR / str(nombre).strip()
    )

    df["nombre_clip"] = df.apply(
        construir_nombre_clip,
        axis=1
    )

    df["ruta_salida"] = df["nombre_clip"].apply(
        lambda nombre: OUTPUT_DIR / nombre
    )

    # -----------------------------------------------------
    # Validaciones
    # -----------------------------------------------------

    df["origen_existe"] = df["ruta_origen"].apply(
        lambda p: p.exists()
    )

    faltantes = df[~df["origen_existe"]]

    duplicados = df[
        df["nombre_clip"].duplicated(keep=False)
    ]

    ya_existentes = df[
        df["ruta_salida"].apply(lambda p: p.exists())
    ]

    print()
    print("VALIDACIÓN")
    print("-" * 80)

    print(f"Clips DCSASS esperados:          {len(df)}")
    print(f"Archivos fuente encontrados:     {df['origen_existe'].sum()}")
    print(f"Archivos fuente faltantes:       {len(faltantes)}")
    print(f"Nombres de salida duplicados:    {len(duplicados)}")
    print(f"Destinos que ya existen:         {len(ya_existentes)}")

    if not faltantes.empty:
        print()
        print("ARCHIVOS FUENTE NO ENCONTRADOS")
        print("-" * 80)

        print(
            faltantes[
                [
                    "video_id",
                    "video_origen",
                    "ruta_origen"
                ]
            ].to_string(index=False)
        )

    if not duplicados.empty:
        print()
        print("NOMBRES DUPLICADOS")
        print("-" * 80)

        print(
            duplicados[
                [
                    "video_id",
                    "clip_id",
                    "nombre_clip"
                ]
            ].to_string(index=False)
        )

    # -----------------------------------------------------
    # Detener si hay errores
    # -----------------------------------------------------

    if not faltantes.empty or not duplicados.empty:
        print()
        print("RESULTADO: VALIDACIÓN NO SUPERADA")
        sys.exit(1)

    # -----------------------------------------------------
    # Mostrar primeros ejemplos
    # -----------------------------------------------------

    print()
    print("PRIMEROS 10 ARCHIVOS")
    print("-" * 80)

    print(
        df[
            [
                "video_id",
                "video_origen",
                "clip_id",
                "nombre_clip"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Dry-run
    # -----------------------------------------------------

    if not args.execute:
        print()
        print("=" * 80)
        print("DRY-RUN COMPLETADO")
        print("No se copió ningún archivo DCSASS.")
        print("=" * 80)
        return

    # -----------------------------------------------------
    # Copia real
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    copiados = 0
    fallidos = 0

    for _, row in df.iterrows():
        origen = row["ruta_origen"]
        destino = row["ruta_salida"]

        try:
            shutil.copy2(
                origen,
                destino
            )

            copiados += 1
            print(f"[OK] {destino.name}")

        except Exception as e:
            fallidos += 1
            print(
                f"[ERROR] {destino.name}: {e}"
            )

    print()
    print("=" * 80)
    print("RESULTADO")
    print("=" * 80)

    print(f"Clips esperados: {len(df)}")
    print(f"Copiados:        {copiados}")
    print(f"Fallidos:        {fallidos}")

    if fallidos:
        sys.exit(1)


if __name__ == "__main__":
    main()