import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


# =========================================================
# Configuración
# =========================================================

MANIFEST_PATH = Path(
    "/opt/workspace/MNA---Proyecto-Integrador/"
    "Datos/Preparacion_Datasets/dataset_manifest_todos.csv"
)

OUTPUT_DIR = Path(
    "/opt/workspace/datasets/procesados/clips_4s"
)

DURACION_OBJETIVO_S = 4.0
TOLERANCIA = 1e-6


# =========================================================
# Resolver ubicación del video original
# =========================================================

def resolver_ruta(row):
    dataset = str(row["dataset"]).strip()
    etiqueta = str(row["etiqueta"]).strip()
    video = str(row["video_origen"]).strip()

    if dataset == "CCTV_YOLO":
        base = Path(
            "/opt/workspace/datasets/CCTV_Shoplifting_Dataset"
        )

    elif dataset == "MNNIT":

        if etiqueta == "ocultamiento":
            base = Path(
                "/opt/workspace/datasets/MNNIT_h264/shoplifting"
            )

        elif etiqueta == "sin_ocultamiento":
            base = Path(
                "/opt/workspace/datasets/MNNIT_h264/normal"
            )

        else:
            return None

    elif dataset == "SINTETICO":
        base = Path(
            "/opt/workspace/datasets/sintetico"
        )

    else:
        return None

    return base / video


# =========================================================
# Construir nombre del clip
# =========================================================

def construir_nombre_clip(row):
    dataset = str(row["dataset"]).strip()
    video_id = str(row["video_id"]).strip()
    clip_id = str(row["clip_id"]).strip()

    # Evitar caracteres problemáticos en nombres de archivo
    video_id = (
        video_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    clip_id = (
        clip_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    return f"{dataset}__{video_id}__{clip_id}.mp4"


# =========================================================
# Generar un clip con FFmpeg
# =========================================================

def generar_clip(origen, destino, inicio_s, duracion_s):
    comando = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",

        "-ss", str(inicio_s),
        "-i", str(origen),

        "-t", str(duracion_s),

        # Solo necesitamos video para los modelos RGB
        "-map", "0:v:0",
        "-an",

        # Recodificar para obtener cortes temporales precisos
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",

        str(destino),
    ]

    subprocess.run(
        comando,
        check=True
    )


# =========================================================
# Programa principal
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera clips RGB de 4 segundos a partir del manifiesto."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Genera físicamente los clips. "
            "Si se omite, solamente realiza dry-run."
        )
    )

    args = parser.parse_args()

    modo = "EJECUCIÓN" if args.execute else "DRY-RUN"

    print("=" * 80)
    print(f"GENERACIÓN DE CLIPS DE 4 SEGUNDOS - {modo}")
    print("=" * 80)

    # -----------------------------------------------------
    # Comprobar FFmpeg
    # -----------------------------------------------------

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        print("ERROR: FFmpeg no está disponible en PATH.")
        sys.exit(1)

    print(f"FFmpeg: {ffmpeg_path}")

    # -----------------------------------------------------
    # Leer manifiesto
    # -----------------------------------------------------

    if not MANIFEST_PATH.exists():
        print(f"ERROR: No existe el manifiesto:")
        print(MANIFEST_PATH)
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)

    columnas_requeridas = {
        "dataset",
        "video_id",
        "video_origen",
        "clip_id",
        "inicio_s",
        "fin_s",
        "etiqueta",
    }

    faltan_columnas = columnas_requeridas - set(df.columns)

    if faltan_columnas:
        print(
            "ERROR: Faltan columnas en el manifiesto:",
            sorted(faltan_columnas)
        )
        sys.exit(1)

    print(f"Registros totales del manifiesto: {len(df)}")

    # -----------------------------------------------------
    # Excluir DCSASS
    # -----------------------------------------------------

    df = df[df["dataset"] != "DCSASS"].copy()

    print(f"Registros después de excluir DCSASS: {len(df)}")

    # -----------------------------------------------------
    # Calcular duración
    #
    # Convención:
    # [inicio_s, fin_s)
    # -----------------------------------------------------

    df["inicio_s"] = pd.to_numeric(
        df["inicio_s"],
        errors="coerce"
    )

    df["fin_s"] = pd.to_numeric(
        df["fin_s"],
        errors="coerce"
    )

    df["duracion_clip_s"] = (
        df["fin_s"] - df["inicio_s"]
    )

    # -----------------------------------------------------
    # Resolver rutas y nombres de salida
    # -----------------------------------------------------

    df["ruta_origen"] = df.apply(
        resolver_ruta,
        axis=1
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

    errores = []

    # Timestamps faltantes
    timestamps_invalidos = df[
        df["inicio_s"].isna()
        | df["fin_s"].isna()
    ]

    if not timestamps_invalidos.empty:
        errores.append(
            f"{len(timestamps_invalidos)} registros "
            "tienen timestamps inválidos."
        )

    # Inicio negativo
    inicio_negativo = df[
        df["inicio_s"] < 0
    ]

    if not inicio_negativo.empty:
        errores.append(
            f"{len(inicio_negativo)} registros "
            "tienen inicio_s negativo."
        )

    # Fin <= inicio
    intervalo_invalido = df[
        df["fin_s"] <= df["inicio_s"]
    ]

    if not intervalo_invalido.empty:
        errores.append(
            f"{len(intervalo_invalido)} registros "
            "tienen fin_s <= inicio_s."
        )

    # Duración diferente de 4 segundos
    duracion_incorrecta = df[
        (
            df["duracion_clip_s"]
            - DURACION_OBJETIVO_S
        ).abs() > TOLERANCIA
    ]

    if not duracion_incorrecta.empty:
        errores.append(
            f"{len(duracion_incorrecta)} registros "
            "no duran exactamente 4 segundos."
        )

    # Ruta no definida
    rutas_no_definidas = df[
        df["ruta_origen"].isna()
    ]

    if not rutas_no_definidas.empty:
        errores.append(
            f"{len(rutas_no_definidas)} registros "
            "no tienen regla de ubicación."
        )

    # Archivo fuente inexistente
    archivos_faltantes = df[
        df["ruta_origen"].apply(
            lambda p: (
                p is None
                or not p.exists()
            )
        )
    ]

    if not archivos_faltantes.empty:
        errores.append(
            f"{len(archivos_faltantes)} registros "
            "no encuentran el video original."
        )

    # Colisiones de nombres
    nombres_duplicados = df[
        df["nombre_clip"].duplicated(
            keep=False
        )
    ]

    if not nombres_duplicados.empty:
        errores.append(
            f"{len(nombres_duplicados)} registros "
            "generarían nombres de clip duplicados."
        )

    # -----------------------------------------------------
    # Mostrar resumen de validación
    # -----------------------------------------------------

    print()
    print("VALIDACIÓN")
    print("-" * 80)

    print(f"Clips a generar:                 {len(df)}")
    print(
        f"Duración diferente de 4 s:      "
        f"{len(duracion_incorrecta)}"
    )
    print(
        f"Rutas sin regla:                "
        f"{len(rutas_no_definidas)}"
    )
    print(
        f"Videos fuente no encontrados:   "
        f"{len(archivos_faltantes)}"
    )
    print(
        f"Nombres de salida duplicados:   "
        f"{len(nombres_duplicados)}"
    )

    # -----------------------------------------------------
    # Detenerse si existen errores
    # -----------------------------------------------------

    if errores:
        print()
        print("ERROR: LA VALIDACIÓN NO FUE SUPERADA")
        print("-" * 80)

        for error in errores:
            print(f"- {error}")

        sys.exit(1)

    print()
    print("VALIDACIÓN SUPERADA")

    # -----------------------------------------------------
    # Mostrar ejemplos
    # -----------------------------------------------------

    print()
    print("PRIMEROS 10 CLIPS")
    print("-" * 80)

    columnas_muestra = [
        "dataset",
        "video_id",
        "clip_id",
        "inicio_s",
        "fin_s",
        "duracion_clip_s",
        "nombre_clip",
    ]

    print(
        df[columnas_muestra]
        .head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Dry-run: detener aquí
    # -----------------------------------------------------

    if not args.execute:
        print()
        print("=" * 80)
        print("DRY-RUN COMPLETADO")
        print("No se generó ningún archivo.")
        print("=" * 80)
        return

    # -----------------------------------------------------
    # Generación real
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    exitosos = 0
    fallidos = 0

    print()
    print("GENERANDO CLIPS")
    print("-" * 80)

    for _, row in df.iterrows():

        origen = row["ruta_origen"]
        destino = row["ruta_salida"]

        inicio_s = row["inicio_s"]
        duracion_s = row["duracion_clip_s"]

        try:
            generar_clip(
                origen=origen,
                destino=destino,
                inicio_s=inicio_s,
                duracion_s=duracion_s,
            )

            exitosos += 1

            print(
                f"[OK] {destino.name}"
            )

        except subprocess.CalledProcessError:
            fallidos += 1

            print(
                f"[ERROR] {destino.name}"
            )

    print()
    print("=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(f"Clips esperados: {len(df)}")
    print(f"Generados:       {exitosos}")
    print(f"Fallidos:        {fallidos}")

    if fallidos > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()