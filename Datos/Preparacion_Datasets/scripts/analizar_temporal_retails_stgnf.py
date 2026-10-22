import json

import numpy as np
import pandas as pd

from validar_retails_stgnf import (
    NOMBRES_PARTICIONES_RETAILS,
    localizar_particiones_retails
)


LONGITUDES_CANDIDATAS = [
    8,
    16,
    24,
    32
]

SEG_LEN_STGNF = 16

STRIDES_CANDIDATOS = [
    1,
    4,
    8,
    16
]


def longitudes_tramos_consecutivos(frame_ids):

    frames = np.asarray(
        sorted(
            set(
                frame_ids
            )
        ),
        dtype=np.int64
    )

    if frames.size == 0:
        return []

    cortes = np.where(
        np.diff(
            frames
        ) != 1
    )[0]

    inicios = np.concatenate(
        [
            np.array(
                [0]
            ),
            cortes + 1
        ]
    )

    finales = np.concatenate(
        [
            cortes,
            np.array(
                [
                    len(
                        frames
                    )
                    - 1
                ]
            )
        ]
    )

    return (
        finales
        - inicios
        + 1
    ).tolist()


def cargar_trayectorias(
    particiones_retails
):

    filas = []

    for nombre_particion in NOMBRES_PARTICIONES_RETAILS:

        ruta_particion = particiones_retails[
            nombre_particion
        ]

        archivos_json = sorted(
            ruta_particion.rglob(
                "*.json"
            )
        )

        for ruta_json in archivos_json:

            with open(
                ruta_json,
                "r",
                encoding="utf-8"
            ) as archivo:

                datos = json.load(
                    archivo
                )

            for persona_id, datos_persona in datos.items():

                frame_ids = sorted(
                    int(
                        frame_id
                    )
                    for frame_id
                    in datos_persona.keys()
                )

                if not frame_ids:
                    continue

                tramos = longitudes_tramos_consecutivos(
                    frame_ids
                )

                span = (
                    frame_ids[-1]
                    - frame_ids[0]
                    + 1
                )

                filas.append(
                    {
                        "particion":
                            nombre_particion,

                        "archivo":
                            ruta_json.stem,

                        "persona_id":
                            str(
                                persona_id
                            ),

                        "observaciones":
                            len(
                                frame_ids
                            ),

                        "huecos":
                            (
                                span
                                - len(
                                    frame_ids
                                )
                            ),

                        "max_run":
                            max(
                                tramos
                            ),

                        "tramos":
                            tramos
                    }
                )

    return pd.DataFrame(
        filas
    )


def resumir_trayectorias(
    trayectorias
):

    filas = []

    for nombre_particion in NOMBRES_PARTICIONES_RETAILS:

        df_particion = trayectorias[
            trayectorias[
                "particion"
            ]
            == nombre_particion
        ]

        fila = {
            "particion":
                nombre_particion,

            "trayectorias":
                len(
                    df_particion
                ),

            "obs_mediana":
                float(
                    df_particion[
                        "observaciones"
                    ].median()
                ),

            "obs_p25":
                float(
                    df_particion[
                        "observaciones"
                    ].quantile(
                        0.25
                    )
                ),

            "obs_p75":
                float(
                    df_particion[
                        "observaciones"
                    ].quantile(
                        0.75
                    )
                ),

            "max_run_mediana":
                float(
                    df_particion[
                        "max_run"
                    ].median()
                ),

            "max_run_p25":
                float(
                    df_particion[
                        "max_run"
                    ].quantile(
                        0.25
                    )
                ),

            "max_run_p75":
                float(
                    df_particion[
                        "max_run"
                    ].quantile(
                        0.75
                    )
                ),

            "trayectorias_con_huecos":
                int(
                    (
                        df_particion[
                            "huecos"
                        ]
                        > 0
                    ).sum()
                )
        }

        for longitud in LONGITUDES_CANDIDATAS:

            aptas = (
                df_particion[
                    "max_run"
                ]
                >= longitud
            )

            fila[
                f"aptas_{longitud}"
            ] = int(
                aptas.sum()
            )

            fila[
                f"pct_aptas_{longitud}"
            ] = float(
                100
                * aptas.mean()
            )

        filas.append(
            fila
        )

    return pd.DataFrame(
        filas
    )


def comparar_strides(
    trayectorias
):

    filas = []

    for nombre_particion in NOMBRES_PARTICIONES_RETAILS:

        df_particion = trayectorias[
            trayectorias[
                "particion"
            ]
            == nombre_particion
        ]

        for stride in STRIDES_CANDIDATOS:

            ventanas = np.asarray(
                [
                    sum(
                        max(
                            0,
                            1
                            + (
                                longitud_tramo
                                - SEG_LEN_STGNF
                            )
                            // stride
                        )
                        for longitud_tramo
                        in tramos
                        if longitud_tramo
                        >= SEG_LEN_STGNF
                    )
                    for tramos
                    in df_particion[
                        "tramos"
                    ]
                ],
                dtype=np.int64
            )

            trayectorias_aptas = (
                ventanas > 0
            )

            ventanas_aptas = ventanas[
                trayectorias_aptas
            ]

            solapamiento = max(
                0.0,
                1.0
                - (
                    stride
                    / SEG_LEN_STGNF
                )
            )

            filas.append(
                {
                    "particion":
                        nombre_particion,

                    "seg_len":
                        SEG_LEN_STGNF,

                    "stride":
                        stride,

                    "solapamiento_pct":
                        100
                        * solapamiento,

                    "trayectorias":
                        len(
                            ventanas
                        ),

                    "trayectorias_con_ventanas":
                        int(
                            trayectorias_aptas.sum()
                        ),

                    "ventanas_totales":
                        int(
                            ventanas.sum()
                        ),

                    "ventanas_mediana_por_trayectoria_apta":
                        (
                            float(
                                np.median(
                                    ventanas_aptas
                                )
                            )
                            if ventanas_aptas.size
                            else 0.0
                        ),

                    "ventanas_media_por_trayectoria_apta":
                        (
                            float(
                                np.mean(
                                    ventanas_aptas
                                )
                            )
                            if ventanas_aptas.size
                            else 0.0
                        )
                }
            )

    return pd.DataFrame(
        filas
    )


def main():

    particiones_retails = (
        localizar_particiones_retails()
    )

    faltantes = [
        nombre
        for nombre
        in NOMBRES_PARTICIONES_RETAILS
        if nombre
        not in particiones_retails
    ]

    if faltantes:
        raise RuntimeError(
            "No se localizaron todas las "
            "particiones de RetailS."
        )

    trayectorias = cargar_trayectorias(
        particiones_retails
    )

    resumen = resumir_trayectorias(
        trayectorias
    )

    comparacion = comparar_strides(
        trayectorias
    )

    print(
        "=== CARACTERIZACIÓN TEMPORAL DE RETAILS ==="
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    print(
        "\n=== COMPARACIÓN DE SEG_STRIDE PARA STG-NF ==="
    )

    print(
        comparacion.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()