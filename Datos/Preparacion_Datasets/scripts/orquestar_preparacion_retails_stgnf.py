import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


try:
    from .construir_cache_retails_stgnf import (
        inventariar_cache,
        comprobar_resumen
    )

except ImportError:
    from construir_cache_retails_stgnf import (
        inventariar_cache,
        comprobar_resumen
    )


VENTANAS_ESPERADAS_STGNF = {
    "RetailS_train": 909689,
    "RetailS_test_staged": 33857,
    "RetailS_test_realworld": 3501
}

TOTAL_VENTANAS_ESPERADO_STGNF = 947047

NPZ_ESPERADOS_STGNF = 1613


def localizar_raiz_repositorio():

    marcador = (
        Path("Datos")
        / "Preparacion_Datasets"
        / "scripts"
        / "validar_retails_stgnf.py"
    )

    for candidato in [
        Path.cwd(),
        *Path.cwd().parents
    ]:

        if (
            candidato
            / marcador
        ).is_file():

            return candidato.resolve()

    raise RuntimeError(
        "No se pudo localizar la raíz "
        "del repositorio."
    )


def obtener_rutas():

    repo_dir = (
        localizar_raiz_repositorio()
    )

    base_preparacion = (
        repo_dir
        / "Datos"
        / "Preparacion_Datasets"
    )

    scripts_dir = (
        base_preparacion
        / "scripts"
    )

    cache_dir = (
        base_preparacion
        / "RetailS_STGNF_cache"
    )

    indice_path = (
        cache_dir
        / "indice_ventanas_stgnf.csv.gz"
    )

    return {
        "repo":
            repo_dir,

        "scripts":
            scripts_dir,

        "cache":
            cache_dir,

        "indice":
            indice_path
    }


def validar_indice():

    rutas = obtener_rutas()

    indice_path = rutas[
        "indice"
    ]

    if not indice_path.is_file():

        return {
            "disponible": False,
            "total": 0,
            "duplicados": None,
            "particiones": {},
            "correcto": False
        }

    indice = pd.read_csv(
        indice_path,
        dtype={
            "particion": str,
            "ruta_json_relativa": str,
            "archivo": str,
            "persona_id": str
        }
    )

    columnas_requeridas = {
        "particion",
        "ruta_json_relativa",
        "archivo",
        "persona_id",
        "frame_inicio",
        "frame_fin",
        "frame_score",
        "stride"
    }

    if not columnas_requeridas.issubset(
        indice.columns
    ):

        return {
            "disponible": True,
            "total": len(indice),
            "duplicados": None,
            "particiones": {},
            "correcto": False
        }

    conteos = (
        indice[
            "particion"
        ]
        .value_counts()
        .to_dict()
    )

    duplicados = int(
        indice.duplicated().sum()
    )

    correcto = (
        len(indice)
        == TOTAL_VENTANAS_ESPERADO_STGNF
        and duplicados == 0
        and all(
            conteos.get(
                particion,
                0
            )
            == esperado

            for particion, esperado
            in VENTANAS_ESPERADAS_STGNF.items()
        )
    )

    return {
        "disponible": True,
        "total": int(
            len(
                indice
            )
        ),
        "duplicados": duplicados,
        "particiones": conteos,
        "correcto": correcto
    }


def obtener_estado():

    resumen_cache = inventariar_cache(
        profundo=False
    )

    cache_correcta = (
        comprobar_resumen(
            resumen_cache
        )
    )

    total_npz = sum(
        datos[
            "validos"
        ]
        for datos
        in resumen_cache.values()
    )

    estado_indice = (
        validar_indice()
    )

    listo = (
        cache_correcta
        and total_npz
        == NPZ_ESPERADOS_STGNF
        and estado_indice[
            "correcto"
        ]
    )

    return {
        "cache":
            resumen_cache,

        "total_npz":
            total_npz,

        "indice":
            estado_indice,

        "listo":
            listo
    }


def mostrar_estado():

    estado = obtener_estado()

    print(
        "=== PREPARACIÓN RETAILS PARA STG-NF ==="
    )

    print(
        "Representación:",
        "COCO18 XY"
    )

    print(
        "Ventana temporal:",
        "16 frames"
    )

    print(
        "Stride entrenamiento:",
        8
    )

    print(
        "Stride evaluación:",
        1
    )

    print()

    print(
        "Ventanas:"
    )

    for particion in [
        "RetailS_train",
        "RetailS_test_staged",
        "RetailS_test_realworld"
    ]:

        print(
            f"  {particion}:",
            estado[
                "indice"
            ][
                "particiones"
            ].get(
                particion,
                0
            )
        )

    print(
        "  Total:",
        estado[
            "indice"
        ][
            "total"
        ]
    )

    print(
        "  Duplicados:",
        estado[
            "indice"
        ][
            "duplicados"
        ]
    )

    print()

    print(
        "Caché NPZ:"
    )

    for particion in [
        "train",
        "staged",
        "realworld"
    ]:

        datos = estado[
            "cache"
        ][
            particion
        ]

        print(
            f"  {particion}:",
            datos[
                "validos"
            ]
        )

    print(
        "  Total:",
        estado[
            "total_npz"
        ]
    )

    print()

    print(
        "Estado:",
        (
            "LISTO PARA STG-NF"
            if estado[
                "listo"
            ]
            else "REVISAR PREPARACIÓN"
        )
    )

    return estado


def ejecutar_script(
    nombre_script
):

    rutas = obtener_rutas()

    script = (
        rutas[
            "scripts"
        ]
        / nombre_script
    )

    if not script.is_file():

        raise FileNotFoundError(
            "No se encontró el script "
            f"{nombre_script}."
        )

    resultado = subprocess.run(
        [
            sys.executable,
            str(
                script
            )
        ],
        cwd=rutas[
            "repo"
        ],
        check=False
    )

    if resultado.returncode != 0:

        raise RuntimeError(
            "Falló la validación "
            f"{nombre_script}."
        )


def ejecutar_validacion_completa():

    scripts_validacion = [
        "validar_retails_stgnf.py",
        "validar_alineacion_retails_stgnf.py",
        "analizar_temporal_retails_stgnf.py",
        "analizar_espacio_coordenadas_retails_stgnf.py",
        "preparar_retails_stgnf.py",
        "construir_cache_retails_stgnf.py"
    ]

    print(
        "=== VALIDACIÓN COMPLETA RETAILS STG-NF ==="
    )

    for nombre_script in scripts_validacion:

        print()
        print(
            "Ejecutando:",
            nombre_script
        )

        if (
            nombre_script
            == "construir_cache_retails_stgnf.py"
        ):

            rutas = obtener_rutas()

            script = (
                rutas[
                    "scripts"
                ]
                / nombre_script
            )

            resultado = subprocess.run(
                [
                    sys.executable,
                    str(
                        script
                    ),
                    "validate"
                ],
                cwd=rutas[
                    "repo"
                ],
                check=False
            )

            if resultado.returncode != 0:

                raise RuntimeError(
                    "Falló la validación "
                    "completa de la caché NPZ."
                )

        else:

            ejecutar_script(
                nombre_script
            )

    print()
    print(
        "=== ESTADO FINAL ==="
    )

    estado = mostrar_estado()

    if not estado[
        "listo"
    ]:

        raise RuntimeError(
            "La preparación de RetailS "
            "no quedó validada."
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Orquestación de la preparación "
            "de RetailS para STG-NF."
        )
    )

    parser.add_argument(
        "accion",
        nargs="?",
        choices=[
            "status",
            "validate"
        ],
        default="status"
    )

    args = parser.parse_args()

    if args.accion == "status":

        estado = mostrar_estado()

        if not estado[
            "listo"
        ]:

            raise RuntimeError(
                "La preparación de RetailS "
                "requiere revisión."
            )

    else:

        ejecutar_validacion_completa()


if __name__ == "__main__":

    main()