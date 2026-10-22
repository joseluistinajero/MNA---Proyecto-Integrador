import json

import numpy as np
import pandas as pd

from validar_retails_stgnf import (
    NOMBRES_PARTICIONES_RETAILS,
    localizar_particiones_retails
)


LIMITE_X_DOCUMENTADO = 1080.0
LIMITE_Y_DOCUMENTADO = 720.0
PASO_MUESTREO_COORDENADAS = 500


def analizar_particion(
    nombre_particion,
    ruta_particion
):

    archivos_json = sorted(
        ruta_particion.rglob("*.json")
    )

    total_puntos = 0

    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    x_negativos = 0
    y_negativos = 0
    x_mayor_1080 = 0
    y_mayor_720 = 0

    muestra_x = []
    muestra_y = []

    contador_observaciones = 0

    for ruta_json in archivos_json:

        with open(
            ruta_json,
            "r",
            encoding="utf-8"
        ) as archivo:
            datos = json.load(archivo)

        for datos_persona in datos.values():

            for observacion in datos_persona.values():

                pose = np.asarray(
                    observacion["keypoints"],
                    dtype=np.float32
                ).reshape(
                    17,
                    3
                )

                x = pose[:, 0]
                y = pose[:, 1]

                if not (
                    np.isfinite(x).all()
                    and np.isfinite(y).all()
                ):
                    raise ValueError(
                        "Se detectaron coordenadas "
                        "x o y no finitas."
                    )

                total_puntos += len(x)

                x_min = min(
                    x_min,
                    float(x.min())
                )

                x_max = max(
                    x_max,
                    float(x.max())
                )

                y_min = min(
                    y_min,
                    float(y.min())
                )

                y_max = max(
                    y_max,
                    float(y.max())
                )

                x_negativos += int(
                    (x < 0).sum()
                )

                y_negativos += int(
                    (y < 0).sum()
                )

                x_mayor_1080 += int(
                    (
                        x
                        > LIMITE_X_DOCUMENTADO
                    ).sum()
                )

                y_mayor_720 += int(
                    (
                        y
                        > LIMITE_Y_DOCUMENTADO
                    ).sum()
                )

                if (
                    contador_observaciones
                    % PASO_MUESTREO_COORDENADAS
                    == 0
                ):
                    muestra_x.extend(
                        x.tolist()
                    )

                    muestra_y.extend(
                        y.tolist()
                    )

                contador_observaciones += 1

    muestra_x = np.asarray(
        muestra_x,
        dtype=np.float32
    )

    muestra_y = np.asarray(
        muestra_y,
        dtype=np.float32
    )

    return {
        "particion":
            nombre_particion,

        "puntos_totales":
            total_puntos,

        "x_min":
            float(x_min),

        "x_p01_aprox":
            float(
                np.percentile(
                    muestra_x,
                    1
                )
            ),

        "x_mediana_aprox":
            float(
                np.percentile(
                    muestra_x,
                    50
                )
            ),

        "x_p99_aprox":
            float(
                np.percentile(
                    muestra_x,
                    99
                )
            ),

        "x_max":
            float(x_max),

        "pct_x_negativo":
            100
            * x_negativos
            / total_puntos,

        "pct_x_mayor_1080":
            100
            * x_mayor_1080
            / total_puntos,

        "y_min":
            float(y_min),

        "y_p01_aprox":
            float(
                np.percentile(
                    muestra_y,
                    1
                )
            ),

        "y_mediana_aprox":
            float(
                np.percentile(
                    muestra_y,
                    50
                )
            ),

        "y_p99_aprox":
            float(
                np.percentile(
                    muestra_y,
                    99
                )
            ),

        "y_max":
            float(y_max),

        "pct_y_negativo":
            100
            * y_negativos
            / total_puntos,

        "pct_y_mayor_720":
            100
            * y_mayor_720
            / total_puntos,

        "muestra_percentiles":
            len(muestra_x)
    }


def main():

    particiones = localizar_particiones_retails()

    faltantes = [
        nombre
        for nombre in NOMBRES_PARTICIONES_RETAILS
        if nombre not in particiones
    ]

    if faltantes:
        raise RuntimeError(
            "Particiones RetailS no localizadas: "
            + ", ".join(faltantes)
        )

    resultados = [
        analizar_particion(
            nombre,
            particiones[nombre]
        )
        for nombre
        in NOMBRES_PARTICIONES_RETAILS
    ]

    tabla = pd.DataFrame(
        resultados
    )

    print(
        "=== ESPACIO DE COORDENADAS DE RETAILS ==="
    )

    print(
        tabla.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()