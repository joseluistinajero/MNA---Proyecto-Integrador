from pathlib import Path

import numpy as np
import pandas as pd


try:
    from .validar_retails_stgnf import (
        NOMBRES_PARTICIONES_RETAILS,
        localizar_particiones_retails
    )

except ImportError:
    from validar_retails_stgnf import (
        NOMBRES_PARTICIONES_RETAILS,
        localizar_particiones_retails
    )


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


def caracterizar_acceso(
    indice,
    particiones
):

    filas_archivos = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
    ):

        ruta_particion = particiones[
            nombre_particion
        ]

        df_particion = indice[
            indice["particion"]
            == nombre_particion
        ]

        ventanas_por_archivo = (
            df_particion
            .groupby(
                "ruta_json_relativa"
            )
            .size()
            .to_dict()
        )

        for ruta_json in sorted(
            ruta_particion.rglob(
                "*.json"
            )
        ):

            ruta_relativa = str(
                ruta_json.relative_to(
                    ruta_particion
                )
            )

            tamano_bytes = int(
                ruta_json.stat().st_size
            )

            filas_archivos.append(
                {
                    "particion":
                        nombre_particion,

                    "ruta_json_relativa":
                        ruta_relativa,

                    "tamano_mb":
                        tamano_bytes
                        / (
                            1024 ** 2
                        ),

                    "ventanas":
                        int(
                            ventanas_por_archivo.get(
                                ruta_relativa,
                                0
                            )
                        )
                }
            )

    return pd.DataFrame(
        filas_archivos
    )


def resumir_acceso(
    caracterizacion
):

    filas_resumen = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
    ):

        df_particion = caracterizacion[
            caracterizacion["particion"]
            == nombre_particion
        ]

        tamanos = df_particion[
            "tamano_mb"
        ].to_numpy()

        ventanas = df_particion[
            "ventanas"
        ].to_numpy()

        filas_resumen.append(
            {
                "particion":
                    nombre_particion,

                "archivos":
                    len(
                        df_particion
                    ),

                "tamano_total_gb":
                    float(
                        tamanos.sum()
                        / 1024
                    ),

                "tamano_mediano_mb":
                    float(
                        np.median(
                            tamanos
                        )
                    ),

                "tamano_p90_mb":
                    float(
                        np.percentile(
                            tamanos,
                            90
                        )
                    ),

                "tamano_p99_mb":
                    float(
                        np.percentile(
                            tamanos,
                            99
                        )
                    ),

                "tamano_max_mb":
                    float(
                        tamanos.max()
                    ),

                "ventanas_mediana":
                    float(
                        np.median(
                            ventanas
                        )
                    ),

                "ventanas_p90":
                    float(
                        np.percentile(
                            ventanas,
                            90
                        )
                    ),

                "ventanas_p99":
                    float(
                        np.percentile(
                            ventanas,
                            99
                        )
                    ),

                "ventanas_max":
                    int(
                        ventanas.max()
                    )
            }
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

    ruta_indice = (
        repo_dir
        / "Datos"
        / "Preparacion_Datasets"
        / "RetailS_STGNF_cache"
        / NOMBRE_ARCHIVO_INDICE
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

    caracterizacion = (
        caracterizar_acceso(
            indice,
            particiones
        )
    )

    resumen = resumir_acceso(
        caracterizacion
    )

    print(
        "=== CARACTERIZACIÓN DE ACCESO A RETAILS ==="
    )

    print(
        resumen.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()