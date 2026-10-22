from pathlib import Path
import os
import json

import numpy as np


# ============================================================
# Configuración
# ============================================================

REPO_DIR = Path(__file__).resolve().parents[3]

NOMBRES_PARTICIONES_RETAILS = [
    "RetailS_train",
    "RetailS_test_staged",
    "RetailS_test_realworld"
]


# ============================================================
# Rutas seguras
# ============================================================

def ruta_segura(ruta):

    ruta = Path(ruta).resolve()

    try:
        return str(
            ruta.relative_to(
                REPO_DIR.resolve()
            )
        )

    except ValueError:
        return (
            f"<fuera_del_repositorio>/{ruta.name}"
        )


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
        NOMBRES_PARTICIONES_RETAILS
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
# Inventario
# ============================================================

def inventariar_retails(particiones):

    resultados = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
    ):

        ruta = particiones.get(
            nombre_particion
        )

        if ruta is None:

            resultados.append(
                {
                    "particion":
                        nombre_particion,

                    "encontrada":
                        False,

                    "ruta":
                        None,

                    "json":
                        0,

                    "npy":
                        0,

                    "otros":
                        0
                }
            )

            continue

        archivos = [
            archivo
            for archivo
            in ruta.rglob("*")
            if archivo.is_file()
        ]

        n_json = sum(
            archivo.suffix.lower()
            == ".json"
            for archivo
            in archivos
        )

        n_npy = sum(
            archivo.suffix.lower()
            == ".npy"
            for archivo
            in archivos
        )

        resultados.append(
            {
                "particion":
                    nombre_particion,

                "encontrada":
                    True,

                "ruta":
                    ruta_segura(
                        ruta
                    ),

                "json":
                    n_json,

                "npy":
                    n_npy,

                "otros":
                    (
                        len(
                            archivos
                        )
                        - n_json
                        - n_npy
                    )
            }
        )

    return resultados


# ============================================================
# Validación estructural
# ============================================================

def resumir_json(ruta_json):

    with open(
        ruta_json,
        "r",
        encoding="utf-8"
    ) as archivo:

        contenido = json.load(
            archivo
        )

    resumen = {
        "tipo_raiz":
            type(
                contenido
            ).__name__,

        "longitud_raiz":
            (
                len(
                    contenido
                )
                if hasattr(
                    contenido,
                    "__len__"
                )
                else None
            ),

        "tipo_primer_elemento":
            None,

        "primeras_claves_raiz":
            None,

        "primeras_claves_elemento":
            None
    }

    if isinstance(
        contenido,
        dict
    ):

        resumen[
            "primeras_claves_raiz"
        ] = list(
            contenido.keys()
        )[:5]

        if contenido:

            primer_valor = next(
                iter(
                    contenido.values()
                )
            )

            resumen[
                "tipo_primer_elemento"
            ] = type(
                primer_valor
            ).__name__

            if isinstance(
                primer_valor,
                dict
            ):

                resumen[
                    "primeras_claves_elemento"
                ] = list(
                    primer_valor.keys()
                )[:5]

    return resumen


def resumir_npy(ruta_npy):

    etiquetas = np.load(
        ruta_npy,
        allow_pickle=False
    )

    valores, conteos = np.unique(
        etiquetas,
        return_counts=True
    )

    distribucion = {
        str(
            valor
        ):
        int(
            conteo
        )
        for valor, conteo
        in zip(
            valores,
            conteos
        )
    }

    return {
        "shape":
            tuple(
                etiquetas.shape
            ),

        "dtype":
            str(
                etiquetas.dtype
            ),

        "valores":
            distribucion
    }


def validar_estructura_retails(
    particiones
):

    resultados = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
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

        archivos_npy = sorted(
            ruta_particion.rglob(
                "*.npy"
            )
        )

        if not archivos_json:

            raise RuntimeError(
                "No se encontraron JSON en "
                f"{nombre_particion}."
            )

        json_ejemplo = (
            archivos_json[0]
        )

        resumen_json = resumir_json(
            json_ejemplo
        )

        npy_ejemplo = None
        emparejado = False

        if archivos_npy:

            candidatos = [
                ruta
                for ruta
                in archivos_npy
                if ruta.stem
                == json_ejemplo.stem
            ]

            if candidatos:

                npy_ejemplo = (
                    candidatos[0]
                )

                emparejado = True

        resumen_npy = (
            resumir_npy(
                npy_ejemplo
            )
            if npy_ejemplo
            is not None
            else None
        )

        resultados.append(
            {
                "particion":
                    nombre_particion,

                "json":
                    str(
                        json_ejemplo.relative_to(
                            ruta_particion
                        )
                    ),

                "tipo_raiz":
                    resumen_json[
                        "tipo_raiz"
                    ],

                "longitud_raiz":
                    resumen_json[
                        "longitud_raiz"
                    ],

                "tipo_primer_elemento":
                    resumen_json[
                        "tipo_primer_elemento"
                    ],

                "claves_raiz":
                    resumen_json[
                        "primeras_claves_raiz"
                    ],

                "claves_elemento":
                    resumen_json[
                        "primeras_claves_elemento"
                    ],

                "npy_emparejado":
                    emparejado,

                "npy":
                    (
                        str(
                            npy_ejemplo.relative_to(
                                ruta_particion
                            )
                        )
                        if npy_ejemplo
                        is not None
                        else None
                    ),

                "npy_shape":
                    (
                        resumen_npy[
                            "shape"
                        ]
                        if resumen_npy
                        else None
                    ),

                "npy_dtype":
                    (
                        resumen_npy[
                            "dtype"
                        ]
                        if resumen_npy
                        else None
                    ),

                "npy_valores":
                    (
                        resumen_npy[
                            "valores"
                        ]
                        if resumen_npy
                        else None
                    )
            }
        )

    return resultados


# ============================================================
# Validación de una observación COCO17
# ============================================================

def validar_observacion_coco17(
    particiones
):

    resultados = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
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

        if not archivos_json:

            raise RuntimeError(
                "No se encontraron JSON en "
                f"{nombre_particion}."
            )

        ruta_json = archivos_json[0]

        with open(
            ruta_json,
            "r",
            encoding="utf-8"
        ) as archivo:

            datos = json.load(
                archivo
            )

        persona_id = next(
            iter(
                datos
            )
        )

        datos_persona = datos[
            persona_id
        ]

        frame_id = next(
            iter(
                datos_persona
            )
        )

        observacion = datos_persona[
            frame_id
        ]

        if not isinstance(
            observacion,
            dict
        ):

            raise TypeError(
                "La observación esperada "
                "debe ser un diccionario."
            )

        if (
            "keypoints"
            not in observacion
        ):

            raise KeyError(
                "La observación no contiene "
                "el campo keypoints."
            )

        keypoints = np.asarray(
            observacion[
                "keypoints"
            ],
            dtype=np.float32
        )

        longitud = int(
            keypoints.size
        )

        if longitud != 51:

            raise ValueError(
                "Se esperaban 51 valores "
                "para COCO17 XYC."
            )

        pose = keypoints.reshape(
            17,
            3
        )

        confianza = pose[
            :,
            2
        ]

        confianza_finita = (
            confianza[
                np.isfinite(
                    confianza
                )
            ]
        )

        conf_min = (
            float(
                confianza_finita.min()
            )
            if confianza_finita.size
            > 0
            else None
        )

        conf_max = (
            float(
                confianza_finita.max()
            )
            if confianza_finita.size
            > 0
            else None
        )

        resultados.append(
            {
                "particion":
                    nombre_particion,

                "archivo":
                    str(
                        ruta_json.relative_to(
                            ruta_particion
                        )
                    ),

                "persona_id":
                    str(
                        persona_id
                    ),

                "frame_id":
                    int(
                        frame_id
                    ),

                "campos":
                    list(
                        observacion.keys()
                    ),

                "valores_keypoints":
                    longitud,

                "shape":
                    tuple(
                        pose.shape
                    ),

                "xy_finitos":
                    bool(
                        np.isfinite(
                            pose[
                                :,
                                :2
                            ]
                        ).all()
                    ),

                "confianza_finita":
                    bool(
                        np.isfinite(
                            confianza
                        ).all()
                    ),

                "conf_min":
                    conf_min,

                "conf_max":
                    conf_max
            }
        )

    return resultados


# ============================================================
# Validación exhaustiva del formato COCO17
# ============================================================

def validar_formato_coco17_completo(
    particiones
):

    resultados = []

    for nombre_particion in (
        NOMBRES_PARTICIONES_RETAILS
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

        total_personas = 0
        total_observaciones = 0

        observaciones_validas_51 = 0
        observaciones_invalidas = 0

        no_finito_x = 0
        no_finito_y = 0
        no_finito_conf = 0

        conf_menor_0 = 0
        conf_mayor_1 = 0

        observaciones_con_no_finitos = 0
        observaciones_conf_mayor_1 = 0

        archivos_con_no_finitos = set()
        archivos_conf_mayor_1 = set()

        conf_max_finita = -np.inf

        for ruta_json in archivos_json:

            with open(
                ruta_json,
                "r",
                encoding="utf-8"
            ) as archivo:

                datos = json.load(
                    archivo
                )

            if not isinstance(
                datos,
                dict
            ):

                raise TypeError(
                    "Raíz JSON inválida en "
                    f"{nombre_particion}."
                )

            total_personas += len(
                datos
            )

            for datos_persona in (
                datos.values()
            ):

                if not isinstance(
                    datos_persona,
                    dict
                ):

                    observaciones_invalidas += 1

                    continue

                for observacion in (
                    datos_persona.values()
                ):

                    total_observaciones += 1

                    if not isinstance(
                        observacion,
                        dict
                    ):

                        observaciones_invalidas += 1

                        continue

                    if (
                        "keypoints"
                        not in observacion
                    ):

                        observaciones_invalidas += 1

                        continue

                    keypoints = np.asarray(
                        observacion[
                            "keypoints"
                        ],
                        dtype=np.float32
                    )

                    if keypoints.size != 51:

                        observaciones_invalidas += 1

                        continue

                    observaciones_validas_51 += 1

                    pose = keypoints.reshape(
                        17,
                        3
                    )

                    x = pose[
                        :,
                        0
                    ]

                    y = pose[
                        :,
                        1
                    ]

                    confianza = pose[
                        :,
                        2
                    ]

                    mascara_x_no_finito = (
                        ~np.isfinite(
                            x
                        )
                    )

                    mascara_y_no_finito = (
                        ~np.isfinite(
                            y
                        )
                    )

                    mascara_conf_no_finito = (
                        ~np.isfinite(
                            confianza
                        )
                    )

                    mascara_conf_menor_0 = (
                        np.isfinite(
                            confianza
                        )
                        & (
                            confianza < 0
                        )
                    )

                    mascara_conf_mayor_1 = (
                        np.isfinite(
                            confianza
                        )
                        & (
                            confianza > 1
                        )
                    )

                    no_finito_x += int(
                        mascara_x_no_finito.sum()
                    )

                    no_finito_y += int(
                        mascara_y_no_finito.sum()
                    )

                    no_finito_conf += int(
                        mascara_conf_no_finito.sum()
                    )

                    conf_menor_0 += int(
                        mascara_conf_menor_0.sum()
                    )

                    conf_mayor_1 += int(
                        mascara_conf_mayor_1.sum()
                    )

                    tiene_no_finitos = bool(
                        mascara_x_no_finito.any()
                        or mascara_y_no_finito.any()
                        or mascara_conf_no_finito.any()
                    )

                    tiene_conf_mayor_1 = bool(
                        mascara_conf_mayor_1.any()
                    )

                    if tiene_no_finitos:

                        observaciones_con_no_finitos += 1

                        archivos_con_no_finitos.add(
                            ruta_json
                        )

                    if tiene_conf_mayor_1:

                        observaciones_conf_mayor_1 += 1

                        archivos_conf_mayor_1.add(
                            ruta_json
                        )

                    confianza_finita = (
                        confianza[
                            np.isfinite(
                                confianza
                            )
                        ]
                    )

                    if (
                        confianza_finita.size
                        > 0
                    ):

                        conf_max_finita = max(
                            conf_max_finita,
                            float(
                                confianza_finita.max()
                            )
                        )

        pct_obs_con_no_finitos = (
            100
            * observaciones_con_no_finitos
            / total_observaciones
            if total_observaciones
            > 0
            else 0.0
        )

        pct_obs_conf_mayor_1 = (
            100
            * observaciones_conf_mayor_1
            / total_observaciones
            if total_observaciones
            > 0
            else 0.0
        )

        resultados.append(
            {
                "particion":
                    nombre_particion,

                "archivos_json":
                    len(
                        archivos_json
                    ),

                "personas":
                    total_personas,

                "observaciones":
                    total_observaciones,

                "observaciones_51":
                    observaciones_validas_51,

                "invalidas":
                    observaciones_invalidas,

                "no_finito_x":
                    no_finito_x,

                "no_finito_y":
                    no_finito_y,

                "no_finito_conf":
                    no_finito_conf,

                "conf_menor_0":
                    conf_menor_0,

                "conf_mayor_1":
                    conf_mayor_1,

                "obs_con_no_finitos":
                    observaciones_con_no_finitos,

                "pct_obs_con_no_finitos":
                    pct_obs_con_no_finitos,

                "archivos_con_no_finitos":
                    len(
                        archivos_con_no_finitos
                    ),

                "obs_conf_mayor_1":
                    observaciones_conf_mayor_1,

                "pct_obs_conf_mayor_1":
                    pct_obs_conf_mayor_1,

                "archivos_conf_mayor_1":
                    len(
                        archivos_conf_mayor_1
                    ),

                "conf_max_finita":
                    (
                        float(
                            conf_max_finita
                        )
                        if np.isfinite(
                            conf_max_finita
                        )
                        else None
                    )
            }
        )

    return resultados


# ============================================================
# Ejecución
# ============================================================

def main():

    particiones = (
        localizar_particiones_retails()
    )

    inventario = (
        inventariar_retails(
            particiones
        )
    )

    print(
        "=== INVENTARIO RETAILS ==="
    )

    for fila in inventario:

        print(
            fila["particion"],
            "| encontrada:",
            fila["encontrada"],
            "| json:",
            fila["json"],
            "| npy:",
            fila["npy"],
            "| otros:",
            fila["otros"]
        )

    encontradas = sum(
        fila["encontrada"]
        for fila
        in inventario
    )

    print(
        "Particiones encontradas:",
        encontradas,
        "de",
        len(
            NOMBRES_PARTICIONES_RETAILS
        )
    )

    if (
        encontradas
        != len(
            NOMBRES_PARTICIONES_RETAILS
        )
    ):

        raise RuntimeError(
            "No se localizaron todas las "
            "particiones requeridas de RetailS."
        )

    estructura = (
        validar_estructura_retails(
            particiones
        )
    )

    print(
        "\n=== ESTRUCTURA RETAILS ==="
    )

    for fila in estructura:

        print(
            fila["particion"]
        )

        print(
            "  JSON:",
            fila["json"]
        )

        print(
            "  raíz:",
            fila["tipo_raiz"],
            "| elementos:",
            fila["longitud_raiz"],
            "| primer elemento:",
            fila["tipo_primer_elemento"]
        )

        print(
            "  claves raíz:",
            fila["claves_raiz"]
        )

        print(
            "  primeros frames:",
            fila["claves_elemento"]
        )

        print(
            "  NPY emparejado:",
            fila["npy_emparejado"]
        )

        if (
            fila["npy"]
            is not None
        ):

            print(
                "  NPY:",
                fila["npy"]
            )

            print(
                "  shape:",
                fila["npy_shape"],
                "| dtype:",
                fila["npy_dtype"],
                "| valores:",
                fila["npy_valores"]
            )

    observaciones = (
        validar_observacion_coco17(
            particiones
        )
    )

    print(
        "\n=== OBSERVACIÓN COCO17 ==="
    )

    for fila in observaciones:

        print(
            fila["particion"]
        )

        print(
            "  archivo:",
            fila["archivo"]
        )

        print(
            "  persona:",
            fila["persona_id"],
            "| frame:",
            fila["frame_id"]
        )

        print(
            "  campos:",
            fila["campos"]
        )

        print(
            "  keypoints:",
            fila["valores_keypoints"],
            "| shape:",
            fila["shape"]
        )

        print(
            "  x/y finitos:",
            fila["xy_finitos"],
            "| confianza finita:",
            fila["confianza_finita"]
        )

        print(
            "  confianza min/max:",
            fila["conf_min"],
            fila["conf_max"]
        )

    validacion_completa = (
        validar_formato_coco17_completo(
            particiones
        )
    )

    print(
        "\n=== VALIDACIÓN COMPLETA COCO17 ==="
    )

    for fila in validacion_completa:

        print(
            fila["particion"]
        )

        print(
            "  archivos:",
            fila["archivos_json"],
            "| personas:",
            fila["personas"],
            "| observaciones:",
            fila["observaciones"]
        )

        print(
            "  observaciones con 51 valores:",
            fila["observaciones_51"],
            "| inválidas:",
            fila["invalidas"]
        )

        print(
            "  no finitos x/y/conf:",
            fila["no_finito_x"],
            fila["no_finito_y"],
            fila["no_finito_conf"]
        )

        print(
            "  confianza < 0:",
            fila["conf_menor_0"],
            "| confianza > 1:",
            fila["conf_mayor_1"]
        )

        print(
            "  observaciones con no finitos:",
            fila["obs_con_no_finitos"],
            "| porcentaje:",
            round(
                fila[
                    "pct_obs_con_no_finitos"
                ],
                6
            ),
            "| archivos:",
            fila[
                "archivos_con_no_finitos"
            ]
        )

        print(
            "  observaciones con confianza > 1:",
            fila[
                "obs_conf_mayor_1"
            ],
            "| porcentaje:",
            round(
                fila[
                    "pct_obs_conf_mayor_1"
                ],
                6
            ),
            "| valores:",
            fila["conf_mayor_1"],
            "| archivos:",
            fila[
                "archivos_conf_mayor_1"
            ]
        )

        print(
            "  confianza máxima finita:",
            fila[
                "conf_max_finita"
            ]
        )


if __name__ == "__main__":
    main()