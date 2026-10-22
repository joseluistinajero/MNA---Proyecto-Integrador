"""
Validación de alineación temporal entre poses JSON
y máscaras NPY de RetailS para el experimento STG-NF.

Este script verifica que los identificadores de frame
presentes en las poses puedan utilizarse como índices
de las máscaras de ground truth.
"""

"""
Validación de alineación temporal entre poses JSON
y máscaras NPY de RetailS para el experimento STG-NF.

Este script verifica que los identificadores de frame
presentes en las poses puedan utilizarse como índices
de las máscaras de ground truth.
"""

from pathlib import Path
import json
import os

import numpy as np


# ============================================================
# Configuración
# ============================================================

REPO_DIR = Path(__file__).resolve().parents[3]

PARTICIONES_PRUEBA = [
    "RetailS_test_staged",
    "RetailS_test_realworld"
]


# ============================================================
# Localización de particiones
# ============================================================

def localizar_particiones_retails():

    raices_busqueda = [
        REPO_DIR,
        REPO_DIR / "Datos",
        REPO_DIR / "data",
        REPO_DIR / "datasets"
    ]

    if "RETAILS_DIR" in os.environ:

        raices_busqueda.insert(
            0,
            Path(
                os.environ["RETAILS_DIR"]
            )
        )

    if "BNEXT_CLIPS_DIR" in os.environ:

        raices_busqueda.append(
            Path(
                os.environ["BNEXT_CLIPS_DIR"]
            ).parent
        )

    raices_unicas = []

    for raiz in raices_busqueda:

        raiz = Path(raiz)

        if raiz not in raices_unicas:

            raices_unicas.append(
                raiz
            )

    particiones = {}

    for nombre_particion in (
        PARTICIONES_PRUEBA
    ):

        coincidencias = []

        for raiz in raices_unicas:

            try:
                existe_raiz = raiz.exists()

            except OSError:
                continue

            if not existe_raiz:
                continue

            candidata_directa = (
                raiz
                / nombre_particion
            )

            try:
                candidata_existe = (
                    candidata_directa.exists()
                )

            except OSError:
                candidata_existe = False

            if candidata_existe:

                coincidencias.append(
                    candidata_directa
                )

            if raiz == REPO_DIR:

                try:

                    coincidencias.extend(
                        [
                            ruta
                            for ruta
                            in raiz.rglob(
                                nombre_particion
                            )
                            if ruta.is_dir()
                        ]
                    )

                except OSError:
                    pass

        coincidencias_unicas = []

        for ruta in coincidencias:

            try:
                ruta_resuelta = (
                    ruta.resolve()
                )

            except OSError:
                continue

            if (
                ruta_resuelta
                not in coincidencias_unicas
            ):

                coincidencias_unicas.append(
                    ruta_resuelta
                )

        if coincidencias_unicas:

            particiones[
                nombre_particion
            ] = coincidencias_unicas[0]

    return particiones


# ============================================================
# Validación temporal
# ============================================================

def validar_alineacion_temporal(
    particiones
):

    resultados = []
    problemas = []

    for nombre_particion in (
        PARTICIONES_PRUEBA
    ):

        if (
            nombre_particion
            not in particiones
        ):

            raise RuntimeError(
                "No se localizó la partición "
                f"{nombre_particion}."
            )

        ruta_particion = particiones[
            nombre_particion
        ]

        archivos_json = sorted(
            ruta_particion.rglob(
                "*.json"
            )
        )

        archivos_npy = {
            ruta.stem: ruta
            for ruta
            in ruta_particion.rglob(
                "*.npy"
            )
        }

        archivos_analizados = 0
        compatibles_indice_directo = 0
        total_frames_fuera = 0

        longitudes_mascara = []
        frames_minimos = []
        frames_maximos = []

        archivos_con_frame_0 = 0
        archivos_con_ultimo_frame = 0

        for ruta_json in archivos_json:

            ruta_npy = archivos_npy.get(
                ruta_json.stem
            )

            if ruta_npy is None:

                problemas.append(
                    {
                        "particion":
                            nombre_particion,

                        "archivo":
                            str(
                                ruta_json.relative_to(
                                    ruta_particion
                                )
                            ),

                        "problema":
                            "sin_npy_correspondiente"
                    }
                )

                continue

            with open(
                ruta_json,
                "r",
                encoding="utf-8"
            ) as archivo:

                datos = json.load(
                    archivo
                )

            mascara = np.load(
                ruta_npy,
                allow_pickle=False
            ).reshape(
                -1
            )

            frame_ids = []
            frame_invalido = False

            for datos_persona in (
                datos.values()
            ):

                if not isinstance(
                    datos_persona,
                    dict
                ):

                    frame_invalido = True

                    problemas.append(
                        {
                            "particion":
                                nombre_particion,

                            "archivo":
                                str(
                                    ruta_json.relative_to(
                                        ruta_particion
                                    )
                                ),

                            "problema":
                                "estructura_persona_invalida"
                        }
                    )

                    continue

                for frame_id in (
                    datos_persona.keys()
                ):

                    try:

                        frame_ids.append(
                            int(
                                frame_id
                            )
                        )

                    except (
                        ValueError,
                        TypeError
                    ):

                        frame_invalido = True

                        problemas.append(
                            {
                                "particion":
                                    nombre_particion,

                                "archivo":
                                    str(
                                        ruta_json.relative_to(
                                            ruta_particion
                                        )
                                    ),

                                "problema":
                                    (
                                        "frame_no_entero:"
                                        f"{frame_id}"
                                    )
                            }
                        )

            if frame_invalido:

                continue

            frames_unicos = np.array(
                sorted(
                    set(
                        frame_ids
                    )
                ),
                dtype=np.int64
            )

            archivos_analizados += 1

            longitudes_mascara.append(
                len(
                    mascara
                )
            )

            if frames_unicos.size == 0:

                problemas.append(
                    {
                        "particion":
                            nombre_particion,

                        "archivo":
                            str(
                                ruta_json.relative_to(
                                    ruta_particion
                                )
                            ),

                        "problema":
                            "sin_frames_pose"
                    }
                )

                continue

            frame_min = int(
                frames_unicos.min()
            )

            frame_max = int(
                frames_unicos.max()
            )

            frames_minimos.append(
                frame_min
            )

            frames_maximos.append(
                frame_max
            )

            fuera = (
                (frames_unicos < 0)
                | (
                    frames_unicos
                    >= len(
                        mascara
                    )
                )
            )

            frames_fuera = int(
                fuera.sum()
            )

            total_frames_fuera += (
                frames_fuera
            )

            if frames_fuera == 0:

                compatibles_indice_directo += 1

            if 0 in frames_unicos:

                archivos_con_frame_0 += 1

            ultimo_frame = (
                len(
                    mascara
                )
                - 1
            )

            if ultimo_frame in frames_unicos:

                archivos_con_ultimo_frame += 1

        resultados.append(
            {
                "particion":
                    nombre_particion,

                "archivos":
                    archivos_analizados,

                "compatibles_indice_directo":
                    compatibles_indice_directo,

                "total_frames_fuera":
                    total_frames_fuera,

                "longitud_mascara_min":
                    (
                        min(
                            longitudes_mascara
                        )
                        if longitudes_mascara
                        else None
                    ),

                "longitud_mascara_max":
                    (
                        max(
                            longitudes_mascara
                        )
                        if longitudes_mascara
                        else None
                    ),

                "frame_min_global":
                    (
                        min(
                            frames_minimos
                        )
                        if frames_minimos
                        else None
                    ),

                "frame_max_global":
                    (
                        max(
                            frames_maximos
                        )
                        if frames_maximos
                        else None
                    ),

                "archivos_con_frame_0":
                    archivos_con_frame_0,

                "archivos_con_ultimo_frame":
                    archivos_con_ultimo_frame,

                "todos_compatibles":
                    (
                        archivos_analizados
                        == compatibles_indice_directo
                        and total_frames_fuera == 0
                    )
            }
        )

    return resultados, problemas


# ============================================================
# Ejecución
# ============================================================

def main():

    particiones = (
        localizar_particiones_retails()
    )

    faltantes = [
        nombre
        for nombre
        in PARTICIONES_PRUEBA
        if nombre
        not in particiones
    ]

    if faltantes:

        raise RuntimeError(
            "No se localizaron todas las "
            "particiones de prueba requeridas: "
            + ", ".join(
                faltantes
            )
        )

    resultados, problemas = (
        validar_alineacion_temporal(
            particiones
        )
    )

    print(
        "=== ALINEACIÓN TEMPORAL RETAILS ==="
    )

    for fila in resultados:

        print(
            fila["particion"]
        )

        print(
            "  archivos:",
            fila["archivos"],
            "| compatibles:",
            fila[
                "compatibles_indice_directo"
            ],
            "| frames fuera:",
            fila[
                "total_frames_fuera"
            ]
        )

        print(
            "  longitud máscara min/max:",
            fila[
                "longitud_mascara_min"
            ],
            fila[
                "longitud_mascara_max"
            ]
        )

        print(
            "  frame global min/max:",
            fila[
                "frame_min_global"
            ],
            fila[
                "frame_max_global"
            ]
        )

        print(
            "  cubren frame 0:",
            fila[
                "archivos_con_frame_0"
            ],
            "| cubren último frame:",
            fila[
                "archivos_con_ultimo_frame"
            ]
        )

        print(
            "  todos compatibles:",
            fila[
                "todos_compatibles"
            ]
        )

    print(
        "Problemas de alineación:",
        len(
            problemas
        )
    )

    if problemas:

        print(
            "\n=== PRIMEROS PROBLEMAS DETECTADOS ==="
        )

        for problema in problemas[:10]:

            print(
                problema
            )


if __name__ == "__main__":
    main()