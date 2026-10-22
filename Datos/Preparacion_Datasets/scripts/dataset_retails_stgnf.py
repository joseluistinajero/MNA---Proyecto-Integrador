import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from collections import OrderedDict

try:
    from .validar_retails_stgnf import (
        localizar_particiones_retails
    )

    from .preparar_retails_stgnf import (
        limpiar_pose_retails,
        coco17_a_coco18_stgnf,
        normalizar_pose_stgnf
    )

except ImportError:
    from validar_retails_stgnf import (
        localizar_particiones_retails
    )

    from preparar_retails_stgnf import (
        limpiar_pose_retails,
        coco17_a_coco18_stgnf,
        normalizar_pose_stgnf
    )


SEG_LEN_STGNF = 16

VID_RES_RETAILS = np.array(
    [
        1080.0,
        1920.0
    ],
    dtype=np.float32
)


class RetailSSTGNFDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        indice,
        particiones,
        seg_len=SEG_LEN_STGNF,
        vid_res=VID_RES_RETAILS,
        model_confidence=False
    ):

        self.indice = (
            indice
            .reset_index(
                drop=True
            )
            .copy()
        )

        self.particiones = particiones

        self.seg_len = int(
            seg_len
        )

        self.vid_res = np.asarray(
            vid_res,
            dtype=np.float32
        )

        self.model_confidence = bool(
            model_confidence
        )


    def __len__(
        self
    ):

        return len(
            self.indice
        )


    def __getitem__(
        self,
        idx
    ):

        fila = self.indice.iloc[
            int(idx)
        ]

        nombre_particion = str(
            fila[
                "particion"
            ]
        )

        ruta_json = (
            self.particiones[
                nombre_particion
            ]
            / str(
                fila[
                    "ruta_json_relativa"
                ]
            )
        )

        with open(
            ruta_json,
            "r",
            encoding="utf-8"
        ) as archivo:
            datos = json.load(
                archivo
            )

        persona_id = str(
            fila[
                "persona_id"
            ]
        )

        datos_persona = datos[
            persona_id
        ]

        frame_inicio = int(
            fila[
                "frame_inicio"
            ]
        )

        frames = range(
            frame_inicio,
            frame_inicio
            + self.seg_len
        )

        poses_coco18 = []

        for frame_id in frames:

            observacion = datos_persona[
                str(frame_id)
            ]

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

            poses_coco18.append(
                pose_coco18
            )

        ventana_tvc = np.stack(
            poses_coco18,
            axis=0
        )

        ventana_ntvc = ventana_tvc[
            None,
            ...
        ]

        ventana_normalizada = (
            normalizar_pose_stgnf(
                ventana_ntvc,
                self.vid_res
            )[0]
        )

        ventana_ctv = np.transpose(
            ventana_normalizada,
            (
                2,
                0,
                1
            )
        )

        if self.model_confidence:
            entrada = ventana_ctv

        else:
            entrada = ventana_ctv[
                :2
            ]

        entrada = np.ascontiguousarray(
            entrada,
            dtype=np.float32
        )

        tensor = torch.from_numpy(
            entrada
        )

        metadata = {
            "particion":
                nombre_particion,

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
                )
        }

        return tensor, metadata

class RetailSSTGNFDatasetCache(
    RetailSSTGNFDataset
):

    def __init__(
        self,
        indice,
        particiones,
        cache_max_archivos=2,
        seg_len=SEG_LEN_STGNF,
        vid_res=VID_RES_RETAILS,
        model_confidence=False
    ):

        super().__init__(
            indice=indice,
            particiones=particiones,
            seg_len=seg_len,
            vid_res=vid_res,
            model_confidence=model_confidence
        )

        self.cache_max_archivos = int(
            cache_max_archivos
        )

        if self.cache_max_archivos < 1:
            raise ValueError(
                "cache_max_archivos debe ser al menos 1."
            )

        self._cache = OrderedDict()

        self.cache_hits = 0
        self.cache_misses = 0


    def _cargar_json(
        self,
        nombre_particion,
        ruta_json_relativa
    ):

        clave = (
            str(nombre_particion),
            str(ruta_json_relativa)
        )

        if clave in self._cache:

            self.cache_hits += 1

            datos = self._cache.pop(
                clave
            )

            self._cache[
                clave
            ] = datos

            return datos

        self.cache_misses += 1

        ruta_json = (
            self.particiones[
                nombre_particion
            ]
            / ruta_json_relativa
        )

        with open(
            ruta_json,
            "r",
            encoding="utf-8"
        ) as archivo:
            datos = json.load(
                archivo
            )

        self._cache[
            clave
        ] = datos

        while (
            len(self._cache)
            > self.cache_max_archivos
        ):
            self._cache.popitem(
                last=False
            )

        return datos


    def __getitem__(
        self,
        idx
    ):

        fila = self.indice.iloc[
            int(idx)
        ]

        nombre_particion = str(
            fila[
                "particion"
            ]
        )

        ruta_json_relativa = str(
            fila[
                "ruta_json_relativa"
            ]
        )

        datos = self._cargar_json(
            nombre_particion,
            ruta_json_relativa
        )

        persona_id = str(
            fila[
                "persona_id"
            ]
        )

        datos_persona = datos[
            persona_id
        ]

        frame_inicio = int(
            fila[
                "frame_inicio"
            ]
        )

        poses_coco18 = []

        for frame_id in range(
            frame_inicio,
            frame_inicio
            + self.seg_len
        ):

            observacion = datos_persona[
                str(frame_id)
            ]

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

            poses_coco18.append(
                pose_coco18
            )

        ventana_ntvc = np.stack(
            poses_coco18,
            axis=0
        )[
            None,
            ...
        ]

        ventana_normalizada = (
            normalizar_pose_stgnf(
                ventana_ntvc,
                self.vid_res
            )[0]
        )

        ventana_ctv = np.transpose(
            ventana_normalizada,
            (
                2,
                0,
                1
            )
        )

        entrada = (
            ventana_ctv
            if self.model_confidence
            else ventana_ctv[:2]
        )

        entrada = np.ascontiguousarray(
            entrada,
            dtype=np.float32
        )

        tensor = torch.from_numpy(
            entrada
        )

        metadata = {
            "particion":
                nombre_particion,

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
                )
        }

        return tensor, metadata

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


def validar_dataset_json():

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

    indice_muestra = (
        indice[
            indice[
                "particion"
            ]
            == "RetailS_test_staged"
        ]
        .iloc[
            [0]
        ]
        .reset_index(
            drop=True
        )
    )

    dataset = RetailSSTGNFDataset(
        indice_muestra,
        particiones
    )


    tensor, metadata = dataset[
        0
    ]

    assert tensor.shape == (
        2,
        16,
        18
    )

    assert tensor.dtype == (
        torch.float32
    )

    assert torch.isfinite(
        tensor
    ).all()

    dataset_cache = (
        RetailSSTGNFDatasetCache(
            indice_muestra,
            particiones,
            cache_max_archivos=2
        )
    )

    tensor_cache, metadata_cache = (
        dataset_cache[0]
    )

    assert torch.equal(
        tensor,
        tensor_cache
    )

    assert (
        metadata
        == metadata_cache
    )

    print(
        "=== VALIDACIÓN DATASET JSON STG-NF ==="
    )

    print(
        "Shape:",
        tuple(
            tensor.shape
        )
    )

    print(
        "Dtype:",
        tensor.dtype
    )

    print(
        "Valores finitos:",
        bool(
            torch.isfinite(
                tensor
            ).all()
        )
    )

    print(
        "Media canal x:",
        float(
            tensor[
                0
            ].mean()
        )
    )

    print(
        "Media canal y:",
        float(
            tensor[
                1
            ].mean()
        )
    )

    print(
        "Std canal y:",
        float(
            tensor[
                1
            ].std(
                unbiased=False
            )
        )
    )

    print(
        "Metadata:",
        metadata
    )


def main():

    validar_dataset_json()


if __name__ == "__main__":
    main()