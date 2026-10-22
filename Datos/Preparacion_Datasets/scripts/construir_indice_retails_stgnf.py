import json
from pathlib import Path

import pandas as pd

from validar_retails_stgnf import (
    NOMBRES_PARTICIONES_RETAILS,
    localizar_particiones_retails
)

from preparar_retails_stgnf import (
    obtener_tramos_consecutivos
)


SEG_LEN_STGNF = 16

CONFIG_STRIDE_RETAILS = {
    "RetailS_train": 8,
    "RetailS_test_staged": 1,
    "RetailS_test_realworld": 1
}

NOMBRE_ARCHIVO_INDICE = (
    "indice_ventanas_stgnf.csv.gz"
)


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


def construir_indice(
    particiones
):

    filas = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
    ):

        ruta_particion = particiones[
            nombre_particion
        ]

        stride = CONFIG_STRIDE_RETAILS[
            nombre_particion
        ]

        for ruta_json in sorted(
            ruta_particion.rglob("*.json")
        ):

            with open(
                ruta_json,
                "r",
                encoding="utf-8"
            ) as archivo:
                datos = json.load(
                    archivo
                )

            ruta_relativa = str(
                ruta_json.relative_to(
                    ruta_particion
                )
            )

            for (
                persona_id,
                datos_persona
            ) in datos.items():

                tramos = (
                    obtener_tramos_consecutivos(
                        datos_persona.keys()
                    )
                )

                for tramo in tramos:

                    if (
                        len(tramo)
                        < SEG_LEN_STGNF
                    ):
                        continue

                    limite = (
                        len(tramo)
                        - SEG_LEN_STGNF
                        + 1
                    )

                    for inicio_local in range(
                        0,
                        limite,
                        stride
                    ):

                        frames = tramo[
                            inicio_local:
                            inicio_local
                            + SEG_LEN_STGNF
                        ]

                        frame_inicio = int(
                            frames[0]
                        )

                        frame_fin = int(
                            frames[-1]
                        )

                        frame_score = int(
                            frame_inicio
                            + SEG_LEN_STGNF
                            // 2
                        )

                        filas.append(
                            (
                                nombre_particion,
                                ruta_relativa,
                                ruta_json.stem,
                                str(persona_id),
                                frame_inicio,
                                frame_fin,
                                frame_score,
                                stride
                            )
                        )

    return pd.DataFrame(
        filas,
        columns=[
            "particion",
            "ruta_json_relativa",
            "archivo",
            "persona_id",
            "frame_inicio",
            "frame_fin",
            "frame_score",
            "stride"
        ]
    )


def validar_indice(
    indice,
    particiones
):

    esperados = {
        "RetailS_train": {
            "ventanas": 909689,
            "archivos": 942,
            "trayectorias": 34697,
            "stride": 8
        },
        "RetailS_test_staged": {
            "ventanas": 33857,
            "archivos": 618,
            "trayectorias": 737,
            "stride": 1
        },
        "RetailS_test_realworld": {
            "ventanas": 3501,
            "archivos": 46,
            "trayectorias": 64,
            "stride": 1
        }
    }

    duplicados = int(
        indice.duplicated(
            subset=[
                "particion",
                "ruta_json_relativa",
                "persona_id",
                "frame_inicio",
                "frame_fin"
            ]
        ).sum()
    )

    if duplicados != 0:
        raise RuntimeError(
            "El índice contiene "
            f"{duplicados} duplicados."
        )

    filas_resumen = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
    ):

        df_particion = indice[
            indice["particion"]
            == nombre_particion
        ].copy()

        trayectoria_id = (
            df_particion[
                "ruta_json_relativa"
            ].astype(str)
            + "::"
            + df_particion[
                "persona_id"
            ].astype(str)
        )

        archivos_totales = len(
            list(
                particiones[
                    nombre_particion
                ].rglob("*.json")
            )
        )

        archivos_con_ventanas = int(
            df_particion[
                "ruta_json_relativa"
            ].nunique()
        )

        resumen = {
            "particion":
                nombre_particion,
            "ventanas":
                len(df_particion),
            "archivos_totales":
                archivos_totales,
            "archivos_con_ventanas":
                archivos_con_ventanas,
            "archivos_sin_ventanas":
                (
                    archivos_totales
                    - archivos_con_ventanas
                ),
            "trayectorias_persona":
                int(
                    trayectoria_id.nunique()
                ),
            "stride":
                int(
                    df_particion[
                        "stride"
                    ].iloc[0]
                )
        }

        esperado = esperados[
            nombre_particion
        ]

        assert (
            resumen["ventanas"]
            == esperado["ventanas"]
        )

        assert (
            resumen[
                "archivos_con_ventanas"
            ]
            == esperado["archivos"]
        )

        assert (
            resumen[
                "trayectorias_persona"
            ]
            == esperado["trayectorias"]
        )

        assert (
            resumen["stride"]
            == esperado["stride"]
        )

        filas_resumen.append(
            resumen
        )

    if len(indice) != 947047:
        raise RuntimeError(
            "El número total de ventanas "
            "no coincide con el esperado."
        )

    return pd.DataFrame(
        filas_resumen
    )


def main():

    repo_dir = (
        localizar_raiz_repositorio()
    )

    particiones = (
        localizar_particiones_retails()
    )

    faltantes = [
        nombre
        for nombre
        in NOMBRES_PARTICIONES_RETAILS
        if nombre not in particiones
    ]

    if faltantes:
        raise RuntimeError(
            "Particiones RetailS "
            "no localizadas: "
            + ", ".join(faltantes)
        )

    cache_dir = (
        repo_dir
        / "Datos"
        / "Preparacion_Datasets"
        / "RetailS_STGNF_cache"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_indice = (
        cache_dir
        / NOMBRE_ARCHIVO_INDICE
    )

    indice = construir_indice(
        particiones
    )

    resumen = validar_indice(
        indice,
        particiones
    )

    indice.to_csv(
        ruta_indice,
        index=False,
        compression="gzip"
    )

    print(
        "=== ÍNDICE STG-NF GENERADO ==="
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    print(
        "\nTotal de ventanas:",
        len(indice)
    )

    print(
        "Duplicados exactos: 0"
    )

    print(
        "Artefacto:",
        ruta_indice.relative_to(
            repo_dir
        )
    )


if __name__ == "__main__":
    main()