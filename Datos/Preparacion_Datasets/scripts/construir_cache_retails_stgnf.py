import argparse
import json
import os
from pathlib import Path

import numpy as np


try:
    from .validar_retails_stgnf import (
        localizar_particiones_retails
    )

    from .preparar_retails_stgnf import (
        limpiar_pose_retails,
        coco17_a_coco18_stgnf
    )

except ImportError:
    from validar_retails_stgnf import (
        localizar_particiones_retails
    )

    from preparar_retails_stgnf import (
        limpiar_pose_retails,
        coco17_a_coco18_stgnf
    )


VERSION_CACHE_STGNF = 1

ARCHIVOS_ESPERADOS_STGNF = {
    "train": 942,
    "staged": 624,
    "realworld": 47
}


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


def obtener_directorio_cache():

    repo_dir = (
        localizar_raiz_repositorio()
    )

    cache_dir = (
        repo_dir
        / "Datos"
        / "Preparacion_Datasets"
        / "RetailS_STGNF_cache"
    )

    if not cache_dir.is_dir():

        raise FileNotFoundError(
            "No se encontró el directorio "
            "Datos/Preparacion_Datasets/"
            "RetailS_STGNF_cache."
        )

    return cache_dir


def obtener_configuracion_cache():

    particiones = (
        localizar_particiones_retails()
    )

    cache_dir = (
        obtener_directorio_cache()
    )

    return {
        "train": {
            "origen":
                particiones[
                    "RetailS_train"
                ],

            "cache":
                cache_dir
                / "train"
        },

        "staged": {
            "origen":
                particiones[
                    "RetailS_test_staged"
                ],

            "cache":
                cache_dir
                / "staged"
        },

        "realworld": {
            "origen":
                particiones[
                    "RetailS_test_realworld"
                ],

            "cache":
                cache_dir
                / "realworld"
        }
    }


def validar_npz_cache_stgnf(
    ruta_npz,
    profundo=False
):

    if not ruta_npz.is_file():

        return (
            False,
            "faltante"
        )

    try:

        with np.load(
            ruta_npz,
            allow_pickle=False
        ) as datos:

            claves_requeridas = {
                "format_version",
                "person_ids",
                "offsets",
                "frame_ids",
                "poses_xy"
            }

            if not claves_requeridas.issubset(
                set(
                    datos.files
                )
            ):

                return (
                    False,
                    "claves"
                )

            version = int(
                np.asarray(
                    datos[
                        "format_version"
                    ]
                ).reshape(
                    -1
                )[0]
            )

            person_ids = np.asarray(
                datos[
                    "person_ids"
                ]
            )

            offsets = np.asarray(
                datos[
                    "offsets"
                ]
            )

            frame_ids = np.asarray(
                datos[
                    "frame_ids"
                ]
            )

            poses_xy = np.asarray(
                datos[
                    "poses_xy"
                ]
            )

            if (
                version
                != VERSION_CACHE_STGNF
            ):

                return (
                    False,
                    "version"
                )

            if offsets.ndim != 1:

                return (
                    False,
                    "offsets_shape"
                )

            if len(
                offsets
            ) != (
                len(
                    person_ids
                )
                + 1
            ):

                return (
                    False,
                    "offsets_personas"
                )

            if (
                len(
                    offsets
                ) == 0
                or int(
                    offsets[0]
                ) != 0
            ):

                return (
                    False,
                    "offset_inicial"
                )

            if not np.all(
                np.diff(
                    offsets
                )
                >= 0
            ):

                return (
                    False,
                    "offsets_orden"
                )

            if int(
                offsets[-1]
            ) != len(
                frame_ids
            ):

                return (
                    False,
                    "offset_final"
                )

            if poses_xy.ndim != 3:

                return (
                    False,
                    "poses_dim"
                )

            if tuple(
                poses_xy.shape[
                    1:
                ]
            ) != (
                18,
                2
            ):

                return (
                    False,
                    "poses_shape"
                )

            if len(
                poses_xy
            ) != len(
                frame_ids
            ):

                return (
                    False,
                    "poses_frames"
                )

            if (
                poses_xy.dtype
                != np.float32
            ):

                return (
                    False,
                    "poses_dtype"
                )

            if not np.issubdtype(
                frame_ids.dtype,
                np.integer
            ):

                return (
                    False,
                    "frames_dtype"
                )

            if not np.issubdtype(
                offsets.dtype,
                np.integer
            ):

                return (
                    False,
                    "offsets_dtype"
                )

            if len(
                np.unique(
                    person_ids
                )
            ) != len(
                person_ids
            ):

                return (
                    False,
                    "personas_duplicadas"
                )

            if profundo:

                if not np.isfinite(
                    poses_xy
                ).all():

                    return (
                        False,
                        "poses_no_finitas"
                    )

    except Exception:

        return (
            False,
            "lectura"
        )

    return (
        True,
        "valido"
    )


def inventariar_cache(
    profundo=False
):

    configuracion = (
        obtener_configuracion_cache()
    )

    resumen = {}

    for particion, rutas in (
        configuracion.items()
    ):

        origen = rutas[
            "origen"
        ]

        cache = rutas[
            "cache"
        ]

        archivos_json = sorted(
            origen.rglob(
                "*.json"
            )
        )

        validos = 0
        faltantes = 0
        incompatibles = 0

        relativos_fuente = set()

        for ruta_json in archivos_json:

            ruta_relativa = (
                ruta_json.relative_to(
                    origen
                )
            )

            ruta_relativa_npz = (
                ruta_relativa.with_suffix(
                    ".npz"
                )
            )

            relativos_fuente.add(
                ruta_relativa_npz
            )

            ruta_npz = (
                cache
                / ruta_relativa_npz
            )

            valido, motivo = (
                validar_npz_cache_stgnf(
                    ruta_npz,
                    profundo=profundo
                )
            )

            if valido:

                validos += 1

            elif motivo == "faltante":

                faltantes += 1

            else:

                incompatibles += 1

        archivos_cache = (
            sorted(
                cache.rglob(
                    "*.npz"
                )
            )
            if cache.is_dir()
            else []
        )

        relativos_cache = {
            ruta.relative_to(
                cache
            )
            for ruta
            in archivos_cache
        }

        huerfanos = len(
            relativos_cache
            - relativos_fuente
        )

        resumen[
            particion
        ] = {
            "json_fuente":
                len(
                    archivos_json
                ),

            "validos":
                validos,

            "faltantes":
                faltantes,

            "incompatibles":
                incompatibles,

            "huerfanos":
                huerfanos
        }

    return resumen


def convertir_json_cache_stgnf(
    ruta_json,
    ruta_npz
):

    with open(
        ruta_json,
        "r",
        encoding="utf-8"
    ) as archivo:

        datos_json = json.load(
            archivo
        )

    person_ids = []
    offsets = [0]
    frame_ids = []
    poses_xy = []

    for persona_id, datos_persona in (
        datos_json.items()
    ):

        frames_ordenados = sorted(
            datos_persona.items(),
            key=lambda elemento: int(
                elemento[0]
            )
        )

        for (
            frame_clave,
            observacion
        ) in frames_ordenados:

            frame_id = int(
                frame_clave
            )

            pose_coco17 = np.asarray(
                observacion[
                    "keypoints"
                ],
                dtype=np.float32
            ).reshape(
                17,
                3
            )

            pose_coco17 = (
                limpiar_pose_retails(
                    pose_coco17
                )
            )

            pose_coco18 = (
                coco17_a_coco18_stgnf(
                    pose_coco17
                )
            )

            pose_xy = np.asarray(
                pose_coco18[
                    :,
                    :2
                ],
                dtype=np.float32
            )

            if not np.isfinite(
                pose_xy
            ).all():

                raise RuntimeError(
                    "Se encontraron coordenadas "
                    "no finitas durante la "
                    "conversión de RetailS."
                )

            frame_ids.append(
                frame_id
            )

            poses_xy.append(
                pose_xy
            )

        person_ids.append(
            str(
                persona_id
            )
        )

        offsets.append(
            len(
                frame_ids
            )
        )

    person_ids = np.asarray(
        person_ids,
        dtype=str
    )

    offsets = np.asarray(
        offsets,
        dtype=np.int64
    )

    frame_ids = np.asarray(
        frame_ids,
        dtype=np.int32
    )

    poses_xy = np.asarray(
        poses_xy,
        dtype=np.float32
    ).reshape(
        -1,
        18,
        2
    )

    ruta_npz.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_temporal = (
        ruta_npz.parent
        / (
            "."
            + ruta_npz.name
            + ".tmp"
        )
    )

    try:

        with open(
            ruta_temporal,
            "wb"
        ) as archivo_temporal:

            np.savez(
                archivo_temporal,

                format_version=np.asarray(
                    [
                        VERSION_CACHE_STGNF
                    ],
                    dtype=np.int32
                ),

                person_ids=person_ids,

                offsets=offsets,

                frame_ids=frame_ids,

                poses_xy=poses_xy
            )

        os.replace(
            ruta_temporal,
            ruta_npz
        )

    finally:

        if ruta_temporal.exists():

            ruta_temporal.unlink()

    valido, motivo = (
        validar_npz_cache_stgnf(
            ruta_npz,
            profundo=True
        )
    )

    if not valido:

        if ruta_npz.exists():

            ruta_npz.unlink()

        raise RuntimeError(
            "El archivo NPZ generado no "
            "superó la validación. "
            f"Motivo: {motivo}"
        )


def construir_cache(
    force_rebuild=False
):

    configuracion = (
        obtener_configuracion_cache()
    )

    tareas = []
    incompatibles = []

    for particion, rutas in (
        configuracion.items()
    ):

        origen = rutas[
            "origen"
        ]

        cache = rutas[
            "cache"
        ]

        for ruta_json in sorted(
            origen.rglob(
                "*.json"
            )
        ):

            ruta_relativa = (
                ruta_json.relative_to(
                    origen
                )
            )

            ruta_npz = (
                cache
                / ruta_relativa.with_suffix(
                    ".npz"
                )
            )

            valido, motivo = (
                validar_npz_cache_stgnf(
                    ruta_npz
                )
            )

            if valido:

                continue

            if (
                motivo != "faltante"
                and not force_rebuild
            ):

                incompatibles.append(
                    (
                        particion,
                        ruta_relativa.as_posix(),
                        motivo
                    )
                )

                continue

            tareas.append(
                (
                    ruta_json,
                    ruta_npz
                )
            )

    if incompatibles:

        raise RuntimeError(
            "Se detectaron archivos NPZ "
            "incompatibles. La construcción "
            "se bloqueó para evitar "
            "sobrescritura automática."
        )

    print(
        "Archivos por generar:",
        len(
            tareas
        )
    )

    for numero, (
        ruta_json,
        ruta_npz
    ) in enumerate(
        tareas,
        start=1
    ):

        convertir_json_cache_stgnf(
            ruta_json,
            ruta_npz
        )

        if (
            numero == 1
            or numero % 100 == 0
            or numero == len(
                tareas
            )
        ):

            print(
                "Progreso:",
                f"{numero}/{len(tareas)}",
                flush=True
            )

    return inventariar_cache(
        profundo=True
    )


def comprobar_resumen(
    resumen
):

    for particion, esperado in (
        ARCHIVOS_ESPERADOS_STGNF.items()
    ):

        datos = resumen[
            particion
        ]

        if (
            datos[
                "json_fuente"
            ]
            != esperado
        ):

            return False

        if (
            datos[
                "validos"
            ]
            != esperado
        ):

            return False

        if datos[
            "faltantes"
        ] != 0:

            return False

        if datos[
            "incompatibles"
        ] != 0:

            return False

        if datos[
            "huerfanos"
        ] != 0:

            return False

    return True


def mostrar_resumen(
    resumen,
    titulo
):

    print(
        f"=== {titulo} ==="
    )

    total_validos = 0

    for particion in [
        "train",
        "staged",
        "realworld"
    ]:

        datos = resumen[
            particion
        ]

        total_validos += datos[
            "validos"
        ]

        print(
            particion,
            {
                "json_fuente":
                    datos[
                        "json_fuente"
                    ],

                "validos":
                    datos[
                        "validos"
                    ],

                "faltantes":
                    datos[
                        "faltantes"
                    ],

                "incompatibles":
                    datos[
                        "incompatibles"
                    ],

                "huerfanos":
                    datos[
                        "huerfanos"
                    ]
            }
        )

    print(
        "Archivos NPZ válidos:",
        total_validos
    )

    print(
        "Caché completa:",
        comprobar_resumen(
            resumen
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Administración protegida de "
            "la caché NPZ RetailS para STG-NF."
        )
    )

    parser.add_argument(
        "accion",
        nargs="?",
        choices=[
            "status",
            "validate",
            "build"
        ],
        default="status"
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true"
    )

    args = parser.parse_args()

    if (
        args.force_rebuild
        and args.accion != "build"
    ):

        raise RuntimeError(
            "--force-rebuild solo puede "
            "utilizarse con build."
        )

    if args.accion == "status":

        resumen = inventariar_cache(
            profundo=False
        )

        mostrar_resumen(
            resumen,
            "ESTADO CACHÉ RETAILS STG-NF"
        )

    elif args.accion == "validate":

        resumen = inventariar_cache(
            profundo=True
        )

        mostrar_resumen(
            resumen,
            "VALIDACIÓN CACHÉ RETAILS STG-NF"
        )

        if not comprobar_resumen(
            resumen
        ):

            raise RuntimeError(
                "La caché NPZ no superó "
                "la validación completa."
            )

    else:

        resumen = construir_cache(
            force_rebuild=(
                args.force_rebuild
            )
        )

        mostrar_resumen(
            resumen,
            "RESULTADO CACHÉ RETAILS STG-NF"
        )

        if not comprobar_resumen(
            resumen
        ):

            raise RuntimeError(
                "La construcción terminó, "
                "pero la caché no quedó completa."
            )


if __name__ == "__main__":

    main()