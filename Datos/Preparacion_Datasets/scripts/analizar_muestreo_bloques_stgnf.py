from pathlib import Path

import numpy as np
import pandas as pd


SEMILLA_MUESTREO_STGNF = 42

TAMANOS_BLOQUE_STGNF = [
    16,
    32,
    64,
    128
]


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


def generar_orden_por_bloques(
    indice,
    tamano_bloque,
    semilla
):

    rng = np.random.default_rng(
        semilla
    )

    bloques = []

    for _, grupo in indice.groupby(
        "ruta_json_relativa",
        sort=False
    ):

        posiciones = (
            grupo.index
            .to_numpy(
                dtype=np.int64
            )
            .copy()
        )

        rng.shuffle(
            posiciones
        )

        for inicio in range(
            0,
            len(posiciones),
            tamano_bloque
        ):

            bloques.append(
                posiciones[
                    inicio:
                    inicio
                    + tamano_bloque
                ]
            )

    orden_bloques = np.arange(
        len(bloques),
        dtype=np.int64
    )

    rng.shuffle(
        orden_bloques
    )

    return np.concatenate(
        [
            bloques[
                bloque_id
            ]
            for bloque_id
            in orden_bloques
        ]
    )


def calcular_rachas(
    archivos_ordenados
):

    if len(
        archivos_ordenados
    ) == 0:

        return np.array(
            [],
            dtype=np.int64
        )

    cambios = np.flatnonzero(
        archivos_ordenados[1:]
        != archivos_ordenados[:-1]
    ) + 1

    limites = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64
            ),
            cambios,
            np.array(
                [
                    len(
                        archivos_ordenados
                    )
                ],
                dtype=np.int64
            )
        ]
    )

    return np.diff(
        limites
    )


def comparar_tamanos_bloque(
    indice_train
):

    archivos_train = (
        indice_train[
            "ruta_json_relativa"
        ]
        .astype(str)
        .to_numpy()
    )

    filas = []

    for tamano_bloque in (
        TAMANOS_BLOQUE_STGNF
    ):

        orden = generar_orden_por_bloques(
            indice_train,
            tamano_bloque,
            SEMILLA_MUESTREO_STGNF
        )

        archivos_ordenados = (
            archivos_train[
                orden
            ]
        )

        transiciones = (
            archivos_ordenados[1:]
            == archivos_ordenados[:-1]
        )

        rachas = calcular_rachas(
            archivos_ordenados
        )

        filas.append(
            {
                "tamano_bloque":
                    tamano_bloque,

                "muestras":
                    len(orden),

                "indices_unicos":
                    int(
                        np.unique(
                            orden
                        ).size
                    ),

                "pct_transiciones_mismo_archivo":
                    100
                    * float(
                        transiciones.mean()
                    ),

                "racha_mediana":
                    float(
                        np.median(
                            rachas
                        )
                    ),

                "racha_p90":
                    float(
                        np.percentile(
                            rachas,
                            90
                        )
                    ),

                "racha_max":
                    int(
                        rachas.max()
                    ),

                "archivos_distintos_primeras_1024":
                    int(
                        np.unique(
                            archivos_ordenados[
                                :1024
                            ]
                        ).size
                    ),

                "archivos_distintos_primeras_8192":
                    int(
                        np.unique(
                            archivos_ordenados[
                                :8192
                            ]
                        ).size
                    )
            }
        )

    return pd.DataFrame(
        filas
    )


def main():

    repo_dir = (
        localizar_raiz_repositorio()
    )

    ruta_indice = (
        repo_dir
        / "Datos"
        / "Preparacion_Datasets"
        / "RetailS_STGNF_cache"
        / "indice_ventanas_stgnf.csv.gz"
    )

    if not ruta_indice.is_file():
        raise FileNotFoundError(
            "No se encontró el índice "
            "de ventanas STG-NF."
        )

    indice = pd.read_csv(
        ruta_indice,
        dtype={
            "particion": str,
            "ruta_json_relativa": str,
            "archivo": str,
            "persona_id": str
        }
    )

    indice_train = (
        indice[
            indice[
                "particion"
            ]
            == "RetailS_train"
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    if len(
        indice_train
    ) != 909689:
        raise RuntimeError(
            "El número de ventanas de "
            "entrenamiento no coincide "
            "con el esperado."
        )

    comparacion = (
        comparar_tamanos_bloque(
            indice_train
        )
    )

    if not (
        comparacion[
            "indices_unicos"
        ]
        == 909689
    ).all():
        raise RuntimeError(
            "Alguna configuración no "
            "cubre todas las ventanas "
            "exactamente una vez."
        )

    print(
        "=== COMPARACIÓN DE MUESTREO POR BLOQUES ==="
    )

    print(
        comparacion.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()