from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


try:
    from .construir_cache_retails_stgnf import (
        obtener_directorio_cache,
        validar_npz_cache_stgnf
    )

    from .dataset_retails_stgnf import (
        RetailSSTGNFDataset
    )

    from .preparar_retails_stgnf import (
        normalizar_pose_stgnf
    )

    from .validar_retails_stgnf import (
        localizar_particiones_retails
    )

except ImportError:
    from construir_cache_retails_stgnf import (
        obtener_directorio_cache,
        validar_npz_cache_stgnf
    )

    from dataset_retails_stgnf import (
        RetailSSTGNFDataset
    )

    from preparar_retails_stgnf import (
        normalizar_pose_stgnf
    )

    from validar_retails_stgnf import (
        localizar_particiones_retails
    )


SEG_LEN_STGNF = 16

VID_RES_RETAILS = np.array(
    [
        1080.0,
        1920.0
    ],
    dtype=np.float32
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


def obtener_particiones_cache():

    cache_dir = (
        obtener_directorio_cache()
    )

    return {
        "RetailS_train": {
            "cache":
                cache_dir
                / "train"
        },

        "RetailS_test_staged": {
            "cache":
                cache_dir
                / "staged"
        },

        "RetailS_test_realworld": {
            "cache":
                cache_dir
                / "realworld"
        }
    }


class RetailSSTGNFDatasetNPZ(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        indice,
        particiones_cache=None,
        cache_max_archivos=2
    ):

        self.indice = (
            indice
            .reset_index(
                drop=True
            )
            .copy()
        )

        if particiones_cache is None:

            particiones_cache = (
                obtener_particiones_cache()
            )

        self.raices_cache = {
            nombre:
                configuracion[
                    "cache"
                ]

            for nombre, configuracion
            in particiones_cache.items()
        }

        self.cache_max_archivos = int(
            cache_max_archivos
        )

        if self.cache_max_archivos < 1:

            raise ValueError(
                "cache_max_archivos debe ser "
                "mayor o igual a 1."
            )

        self._cache = OrderedDict()

        self.cache_hits = 0
        self.cache_misses = 0


    def __len__(
        self
    ):

        return len(
            self.indice
        )


    def limpiar_cache(
        self
    ):

        self._cache.clear()

        self.cache_hits = 0
        self.cache_misses = 0


    def _ruta_npz(
        self,
        fila
    ):

        particion = str(
            fila[
                "particion"
            ]
        )

        if particion not in self.raices_cache:

            raise KeyError(
                "La partición indicada no está "
                "configurada en la caché STG-NF."
            )

        ruta_relativa = Path(
            str(
                fila[
                    "ruta_json_relativa"
                ]
            )
        ).with_suffix(
            ".npz"
        )

        return (
            self.raices_cache[
                particion
            ]
            / ruta_relativa
        )


    def _cargar_archivo(
        self,
        ruta_npz
    ):

        clave_cache = str(
            ruta_npz
        )

        if clave_cache in self._cache:

            self.cache_hits += 1

            datos = self._cache.pop(
                clave_cache
            )

            self._cache[
                clave_cache
            ] = datos

            return datos

        self.cache_misses += 1

        valido, motivo = (
            validar_npz_cache_stgnf(
                ruta_npz
            )
        )

        if not valido:

            raise RuntimeError(
                "Se intentó utilizar un archivo "
                "NPZ no válido. "
                f"Motivo: {motivo}"
            )

        with np.load(
            ruta_npz,
            allow_pickle=False
        ) as archivo_npz:

            person_ids = (
                archivo_npz[
                    "person_ids"
                ].copy()
            )

            offsets = (
                archivo_npz[
                    "offsets"
                ].copy()
            )

            frame_ids = (
                archivo_npz[
                    "frame_ids"
                ].copy()
            )

            poses_xy = (
                archivo_npz[
                    "poses_xy"
                ].copy()
            )

        indice_personas = {
            str(
                persona_id
            ):
                posicion

            for posicion, persona_id
            in enumerate(
                person_ids
            )
        }

        datos = {
            "person_ids":
                person_ids,

            "indice_personas":
                indice_personas,

            "offsets":
                offsets,

            "frame_ids":
                frame_ids,

            "poses_xy":
                poses_xy
        }

        self._cache[
            clave_cache
        ] = datos

        while len(
            self._cache
        ) > self.cache_max_archivos:

            self._cache.popitem(
                last=False
            )

        return datos


    def __getitem__(
        self,
        indice
    ):

        fila = self.indice.iloc[
            indice
        ]

        ruta_npz = self._ruta_npz(
            fila
        )

        datos = self._cargar_archivo(
            ruta_npz
        )

        persona_id = str(
            fila[
                "persona_id"
            ]
        )

        if persona_id not in (
            datos[
                "indice_personas"
            ]
        ):

            raise RuntimeError(
                "La trayectoria solicitada no "
                "existe en el archivo NPZ."
            )

        posicion_persona = (
            datos[
                "indice_personas"
            ][
                persona_id
            ]
        )

        inicio_persona = int(
            datos[
                "offsets"
            ][
                posicion_persona
            ]
        )

        fin_persona = int(
            datos[
                "offsets"
            ][
                posicion_persona
                + 1
            ]
        )

        frames_persona = (
            datos[
                "frame_ids"
            ][
                inicio_persona:
                fin_persona
            ]
        )

        poses_persona = (
            datos[
                "poses_xy"
            ][
                inicio_persona:
                fin_persona
            ]
        )

        frame_inicio = int(
            fila[
                "frame_inicio"
            ]
        )

        frames_objetivo = np.arange(
            frame_inicio,
            frame_inicio
            + SEG_LEN_STGNF,
            dtype=np.int32
        )

        posiciones = np.searchsorted(
            frames_persona,
            frames_objetivo
        )

        if np.any(
            posiciones
            >= len(
                frames_persona
            )
        ):

            raise RuntimeError(
                "La ventana solicitada excede "
                "la trayectoria disponible."
            )

        if not np.array_equal(
            frames_persona[
                posiciones
            ],
            frames_objetivo
        ):

            raise RuntimeError(
                "La trayectoria no contiene "
                "de forma continua todos los "
                "frames de la ventana solicitada."
            )

        ventana_xy = (
            poses_persona[
                posiciones
            ]
        )

        ventana_ntvc = np.zeros(
            (
                1,
                SEG_LEN_STGNF,
                18,
                3
            ),
            dtype=np.float32
        )

        ventana_ntvc[
            0,
            ...,
            :2
        ] = ventana_xy

        ventana_normalizada = (
            normalizar_pose_stgnf(
                ventana_ntvc,
                VID_RES_RETAILS
            )[
                0,
                ...,
                :2
            ]
        )

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                np.transpose(
                    ventana_normalizada,
                    (
                        2,
                        0,
                        1
                    )
                ),
                dtype=np.float32
            )
        )

        metadata = {
            "particion":
                str(
                    fila[
                        "particion"
                    ]
                ),

            "archivo":
                str(
                    fila[
                        "archivo"
                    ]
                ),

            "persona_id":
                persona_id,

            "frame_inicio":
                frame_inicio,

            "frame_fin":
                int(
                    fila[
                        "frame_fin"
                    ]
                ),

            "frame_score":
                int(
                    fila[
                        "frame_score"
                    ]
                ),

            "ruta_json_relativa":
                str(
                    fila[
                        "ruta_json_relativa"
                    ]
                )
        }

        return (
            tensor,
            metadata
        )


def cargar_indice():

    repo_dir = (
        localizar_raiz_repositorio()
    )

    indice_path = (
        repo_dir
        / "Datos"
        / "Preparacion_Datasets"
        / "RetailS_STGNF_cache"
        / "indice_ventanas_stgnf.csv.gz"
    )

    if not indice_path.is_file():

        raise FileNotFoundError(
            "No se encontró el índice "
            "de ventanas STG-NF."
        )

    return pd.read_csv(
        indice_path,
        dtype={
            "particion": str,
            "ruta_json_relativa": str,
            "archivo": str,
            "persona_id": str
        }
    )


def validar_equivalencia_json_npz():

    indice = cargar_indice()

    filas_validacion = (
        indice
        .groupby(
            "particion",
            sort=True,
            group_keys=False
        )
        .head(
            1
        )
        .reset_index(
            drop=True
        )
    )

    particiones_retails = (
        localizar_particiones_retails()
    )

    dataset_json = (
        RetailSSTGNFDataset(
            filas_validacion,
            particiones_retails
        )
    )

    dataset_npz = (
        RetailSSTGNFDatasetNPZ(
            filas_validacion,
            cache_max_archivos=4
        )
    )

    resultados = []

    for indice_muestra in range(
        len(
            filas_validacion
        )
    ):

        tensor_json, meta_json = (
            dataset_json[
                indice_muestra
            ]
        )

        tensor_npz, meta_npz = (
            dataset_npz[
                indice_muestra
            ]
        )

        claves_metadata_comunes = [
            "particion",
            "archivo",
            "persona_id",
            "frame_inicio",
            "frame_fin",
            "frame_score"
        ]

        metadata_coincidente = all(
            meta_json[
                clave
            ]
            == meta_npz[
                clave
            ]

            for clave
            in claves_metadata_comunes
        )

        if not metadata_coincidente:

            raise RuntimeError(
                "Los metadatos comunes de la "
                "ventana JSON y NPZ no coinciden."
            )
        diferencia_maxima = float(
            torch.max(
                torch.abs(
                    tensor_json
                    - tensor_npz
                )
            )
        )

        resultados.append(
            {
                "particion":
                    meta_npz[
                        "particion"
                    ],

                "shape_json":
                    tuple(
                        tensor_json.shape
                    ),

                "shape_npz":
                    tuple(
                        tensor_npz.shape
                    ),

                "diferencia_maxima":
                    diferencia_maxima,

                "equivalente":
                    bool(
                        torch.allclose(
                            tensor_json,
                            tensor_npz,
                            atol=1e-6,
                            rtol=1e-6
                        )
                    ),

                "finito":
                    bool(
                        torch.isfinite(
                            tensor_npz
                        ).all()
                    )
            }
        )

    return pd.DataFrame(
        resultados
    )


def main():

    tabla = (
        validar_equivalencia_json_npz()
    )

    print(
        "=== EQUIVALENCIA JSON → NPZ STG-NF ==="
    )

    print(
        tabla.to_string(
            index=False
        )
    )

    todas_equivalentes = bool(
        tabla[
            "equivalente"
        ].all()
    )

    todos_finitos = bool(
        tabla[
            "finito"
        ].all()
    )

    diferencia_maxima = float(
        tabla[
            "diferencia_maxima"
        ].max()
    )

    print()

    print(
        "Todas equivalentes:",
        todas_equivalentes
    )

    print(
        "Todos finitos:",
        todos_finitos
    )

    print(
        "Diferencia máxima global:",
        diferencia_maxima
    )

    if not (
        todas_equivalentes
        and todos_finitos
    ):

        raise RuntimeError(
            "La representación NPZ no es "
            "equivalente a la referencia JSON."
        )


if __name__ == "__main__":

    main()