import argparse
import shutil
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


ZIP_PATH = Path(
    "/opt/workspace/datasets/"
    "Anomaly Detection Dataset UCF/"
    "Shoplifting_Clips.zip"
)

DESTINO = Path(
    "/opt/workspace/datasets/"
    "Anomaly Detection Dataset UCF/"
    "Shoplifting_Clips"
)


def normalizar(nombre):
    """Convierte separadores Windows del ZIP a separadores POSIX."""
    return nombre.replace("\\", "/")


def validar_destino():
    mp4_archivos = sorted(
        p for p in DESTINO.rglob("*.mp4")
        if p.is_file()
    )

    carpetas_raiz = sorted(
        p for p in DESTINO.iterdir()
        if p.is_dir()
    )

    mp4_raiz = sorted(
        p for p in DESTINO.glob("*.mp4")
        if p.is_file()
    )

    directorios_mp4 = sorted(
        p for p in DESTINO.rglob("*.mp4")
        if p.is_dir()
    )

    conteos = {
        carpeta.name: len([
            p for p in carpeta.glob("*.mp4")
            if p.is_file()
        ])
        for carpeta in carpetas_raiz
    }

    carpetas_incorrectas = {
        nombre: cantidad
        for nombre, cantidad in conteos.items()
        if cantidad != 32
    }

    print("=" * 80)
    print("VALIDACIÓN DEL DATASET EXTRAÍDO")
    print("=" * 80)

    print(f"Archivos MP4 reales:            {len(mp4_archivos)}")
    print(f"Carpetas fuente:                {len(carpetas_raiz)}")
    print(f"Directorios terminados en .mp4: {len(directorios_mp4)}")
    print(f"Archivos MP4 en raíz:           {len(mp4_raiz)}")
    print(f"Carpetas distintas de 32 MP4:   {len(carpetas_incorrectas)}")

    if carpetas_incorrectas:
        print()
        print("CARPETAS CON CONTEO INESPERADO")
        print("-" * 80)

        for nombre, cantidad in carpetas_incorrectas.items():
            print(f"{nombre}: {cantidad}")

    print()

    valido = (
        len(mp4_archivos) == 896
        and len(carpetas_raiz) == 28
        and len(mp4_raiz) == 0
        and len(carpetas_incorrectas) == 0
    )

    if valido:
        print("RESULTADO: EXTRACCIÓN VALIDADA")
        return True

    print("RESULTADO: REVISIÓN NECESARIA")
    return False


def extraer():
    if not ZIP_PATH.exists():
        print(f"ERROR: no existe {ZIP_PATH}")
        return False

    DESTINO.mkdir(parents=True, exist_ok=True)

    existentes = [
        p for p in DESTINO.rglob("*.mp4")
        if p.is_file()
    ]

    if existentes:
        print(
            f"ERROR: la carpeta destino ya contiene "
            f"{len(existentes)} archivos MP4."
        )
        print(
            "Use --validate-only para validar "
            "la extracción existente."
        )
        return False

    with ZipFile(ZIP_PATH) as z:
        miembros = []

        for info in z.infolist():
            nombre = normalizar(info.filename)

            if not nombre.lower().endswith(".mp4"):
                continue

            ruta = PurePosixPath(nombre)

            if len(ruta.parts) != 2:
                print(
                    "ERROR: estructura inesperada:",
                    nombre
                )
                return False

            if ".." in ruta.parts:
                print(
                    "ERROR: ruta insegura:",
                    nombre
                )
                return False

            miembros.append((info, ruta))

        carpetas = {
            ruta.parts[0]
            for _, ruta in miembros
        }

        print("=" * 80)
        print("VALIDACIÓN PREVIA")
        print("=" * 80)

        print(f"MP4 a extraer:               {len(miembros)}")
        print(f"Carpetas fuente detectadas:  {len(carpetas)}")

        if len(miembros) != 896:
            print("ERROR: se esperaban 896 MP4.")
            return False

        if len(carpetas) != 28:
            print("ERROR: se esperaban 28 carpetas fuente.")
            return False

        for info, ruta in miembros:
            destino = DESTINO.joinpath(*ruta.parts)

            destino.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with z.open(info) as origen:
                with open(destino, "wb") as salida:
                    shutil.copyfileobj(
                        origen,
                        salida
                    )

    print()
    return validar_destino()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrae y valida los clips DCSASS."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Valida los archivos ya extraídos "
            "sin volver a extraer el ZIP."
        )
    )

    args = parser.parse_args()

    if args.validate_only:
        if not DESTINO.exists():
            print("ERROR: no existe la carpeta destino.")
            sys.exit(1)

        correcto = validar_destino()

    else:
        correcto = extraer()

    if not correcto:
        sys.exit(1)


if __name__ == "__main__":
    main()