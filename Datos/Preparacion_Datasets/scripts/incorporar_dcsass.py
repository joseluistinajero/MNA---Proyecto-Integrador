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
    "/opt/workspace/datasets/"
    "Anomaly Detection Dataset UCF/"
    "Shoplifting_Clips"
)

OUTPUT_DIR = Path(
    "/opt/workspace/datasets/procesados/clips_4s"
)

ETIQUETAS_MODELO = {
    "ocultamiento",
    "sin_ocultamiento",
}


def limpiar(valor):
    return (
        str(valor)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def construir_nombre_salida(row):
    return (
        f"{limpiar(row['dataset'])}__"
        f"{limpiar(row['video_id'])}__"
        f"{limpiar(row['clip_id'])}.mp4"
    )


def construir_ruta_origen(row):
    carpeta = f"{str(row['video_id']).strip()}.mp4"
    archivo = str(row["video_origen"]).strip()

    return DCSASS_DIR / carpeta / archivo


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Selecciona e incorpora clips DCSASS "
            "al dataset operativo clips_4s."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Realiza la copia. Sin esta opción solo hace dry-run."
    )

    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print("ERROR: no existe el manifiesto.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)

    dcsass = df[
        df["dataset"]
        .astype(str)
        .str.strip()
        .eq("DCSASS")
    ].copy()

    dcsass["etiqueta"] = (
        dcsass["etiqueta"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Validar corrección temporal
    # -----------------------------------------------------

    inicio = pd.to_numeric(
        dcsass["inicio_s"],
        errors="coerce"
    )

    fin = pd.to_numeric(
        dcsass["fin_s"],
        errors="coerce"
    )

    patron_temporal_correcto = (
        inicio.eq(0)
        & fin.eq(4)
    )

    # -----------------------------------------------------
    # Selección binaria
    # -----------------------------------------------------

    seleccion = dcsass[
        dcsass["etiqueta"].isin(ETIQUETAS_MODELO)
    ].copy()

    excluidos = dcsass[
        ~dcsass["etiqueta"].isin(ETIQUETAS_MODELO)
    ].copy()

    # -----------------------------------------------------
    # Construir rutas
    # -----------------------------------------------------

    seleccion["ruta_origen"] = seleccion.apply(
        construir_ruta_origen,
        axis=1
    )

    seleccion["nombre_salida"] = seleccion.apply(
        construir_nombre_salida,
        axis=1
    )

    seleccion["ruta_salida"] = seleccion[
        "nombre_salida"
    ].apply(
        lambda nombre: OUTPUT_DIR / nombre
    )

    seleccion["origen_existe"] = seleccion[
        "ruta_origen"
    ].apply(
        lambda p: p.is_file()
    )

    seleccion["destino_existe"] = seleccion[
        "ruta_salida"
    ].apply(
        lambda p: p.exists()
    )

    # -----------------------------------------------------
    # Validaciones
    # -----------------------------------------------------

    faltantes = seleccion[
        ~seleccion["origen_existe"]
    ]

    destinos_existentes = seleccion[
        seleccion["destino_existe"]
    ]

    nombres_duplicados = seleccion[
        seleccion["nombre_salida"].duplicated(
            keep=False
        )
    ]

    print("=" * 80)
    print("INCORPORACIÓN DCSASS A clips_4s")
    print("=" * 80)

    print(f"Registros DCSASS:                 {len(dcsass)}")
    print(
        f"DCSASS con patrón temporal 0-4:  "
        f"{patron_temporal_correcto.sum()}"
    )
    print(f"Clips elegibles:                  {len(seleccion)}")
    print(f"Clips excluidos:                  {len(excluidos)}")

    print()
    print("ETIQUETAS ELEGIBLES")
    print("-" * 80)

    print(
        seleccion["etiqueta"]
        .value_counts()
        .to_string()
    )

    print()
    print("ETIQUETAS EXCLUIDAS")
    print("-" * 80)

    if excluidos.empty:
        print("Ninguna")
    else:
        print(
            excluidos["etiqueta"]
            .value_counts()
            .to_string()
        )

    print()
    print("VALIDACIÓN")
    print("-" * 80)

    print(f"Archivos fuente faltantes:       {len(faltantes)}")
    print(f"Nombres de salida duplicados:    {len(nombres_duplicados)}")
    print(f"Destinos que ya existen:         {len(destinos_existentes)}")

    # -----------------------------------------------------
    # Condiciones obligatorias
    # -----------------------------------------------------

    if len(dcsass) != 784:
        print("ERROR: se esperaban 784 registros DCSASS.")
        sys.exit(1)

    if patron_temporal_correcto.sum() != 784:
        print(
            "ERROR: los 784 registros DCSASS "
            "deben tener inicio_s=0 y fin_s=4."
        )
        sys.exit(1)

    if len(seleccion) != 783:
        print("ERROR: se esperaban 783 clips elegibles.")
        sys.exit(1)

    if not faltantes.empty:
        print()
        print("ARCHIVOS FALTANTES")
        print("-" * 80)

        print(
            faltantes[
                [
                    "video_id",
                    "video_origen",
                    "ruta_origen",
                ]
            ].to_string(index=False)
        )

        sys.exit(1)

    if not nombres_duplicados.empty:
        print("ERROR: existen nombres de salida duplicados.")
        sys.exit(1)

    if not destinos_existentes.empty:
        print(
            "ERROR: algunos archivos DCSASS "
            "ya existen en clips_4s."
        )
        sys.exit(1)

    # -----------------------------------------------------
    # Ejemplos
    # -----------------------------------------------------

    print()
    print("PRIMEROS 10 CLIPS A COPIAR")
    print("-" * 80)

    print(
        seleccion[
            [
                "video_id",
                "video_origen",
                "clip_id",
                "etiqueta",
                "nombre_salida",
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
        print("No se copió ningún archivo.")
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

    for _, row in seleccion.iterrows():
        try:
            shutil.copy2(
                row["ruta_origen"],
                row["ruta_salida"]
            )
            copiados += 1

        except Exception as e:
            fallidos += 1
            print(
                f"[ERROR] {row['nombre_salida']}: {e}"
            )

    print()
    print("=" * 80)
    print("RESULTADO")
    print("=" * 80)

    print(f"Esperados: {len(seleccion)}")
    print(f"Copiados:  {copiados}")
    print(f"Fallidos:  {fallidos}")

    if fallidos:
        sys.exit(1)


if __name__ == "__main__":
    main()