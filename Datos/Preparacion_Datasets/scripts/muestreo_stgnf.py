from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEMILLA_MUESTREO_STGNF = 42
TAMANO_BLOQUE_STGNF = 16


class MuestreadorBloquesArchivo(
    torch.utils.data.Sampler
):

    def __init__(
        self,
        indice,
        tamano_bloque=TAMANO_BLOQUE_STGNF,
        semilla=SEMILLA_MUESTREO_STGNF
    ):

        self.indice = (
            indice
            .reset_index(
                drop=True
            )
        )

        self.tamano_bloque = int(
            tamano_bloque
        )

        self.semilla = int(
            semilla
        )

        self.epoch = 0

        if self.tamano_bloque < 1:
            raise ValueError(
                "tamano_bloque debe ser al menos 1."
            )

        self.grupos_archivo = []

        for _, grupo in self.indice.groupby(
            "ruta_json_relativa",
            sort=False
        ):

            self.grupos_archivo.append(
                grupo.index.to_numpy(
                    dtype=np.int64
                )
            )


    def __len__(
        self
    ):

        return len(
            self.indice
        )


    def set_epoch(
        self,
        epoch
    ):

        self.epoch = int(
            epoch
        )


    def __iter__(
        self
    ):

        rng = np.random.default_rng(
            self.semilla
            + self.epoch
        )

        bloques = []

        for posiciones_originales in (
            self.grupos_archivo
        ):

            posiciones = (
                posiciones_originales
                .copy()
            )

            rng.shuffle(
                posiciones
            )

            for inicio in range(
                0,
                len(
                    posiciones
                ),
                self.tamano_bloque
            ):

                bloques.append(
                    posiciones[
                        inicio:
                        inicio
                        + self.tamano_bloque
                    ]
                )

        orden_bloques = np.arange(
            len(
                bloques
            ),
            dtype=np.int64
        )

        rng.shuffle(
            orden_bloques
        )

        for bloque_id in orden_bloques:

            for indice_muestra in bloques[
                bloque_id
            ]:

                yield int(
                    indice_muestra
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


def obtener_orden(
    sampler,
    epoch
):

    sampler.set_epoch(
        epoch
    )

    return np.fromiter(
        iter(
            sampler
        ),
        dtype=np.int64,
        count=len(
            sampler
        )
    )


def validar_sampler():

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

    sampler = MuestreadorBloquesArchivo(
        indice_train
    )

    orden_epoch_0 = obtener_orden(
        sampler,
        0
    )

    orden_epoch_1 = obtener_orden(
        sampler,
        1
    )

    sampler_repetido = (
        MuestreadorBloquesArchivo(
            indice_train
        )
    )

    orden_epoch_0_repetido = (
        obtener_orden(
            sampler_repetido,
            0
        )
    )

    total = len(
        indice_train
    )

    unicos_0 = np.unique(
        orden_epoch_0
    ).size

    unicos_1 = np.unique(
        orden_epoch_1
    ).size

    cobertura_0 = (
        unicos_0 == total
        and orden_epoch_0.min() == 0
        and orden_epoch_0.max()
        == total - 1
    )

    cobertura_1 = (
        unicos_1 == total
        and orden_epoch_1.min() == 0
        and orden_epoch_1.max()
        == total - 1
    )

    reproducible = np.array_equal(
        orden_epoch_0,
        orden_epoch_0_repetido
    )

    orden_diferente = not np.array_equal(
        orden_epoch_0,
        orden_epoch_1
    )

    archivos = (
        indice_train[
            "ruta_json_relativa"
        ]
        .astype(str)
        .to_numpy()
    )

    archivos_ordenados = archivos[
        orden_epoch_0
    ]

    pct_mismo_archivo = (
        100
        * float(
            (
                archivos_ordenados[1:]
                == archivos_ordenados[:-1]
            ).mean()
        )
    )

    assert cobertura_0
    assert cobertura_1
    assert reproducible
    assert orden_diferente

    print(
        "=== VALIDACIÓN DEL SAMPLER STG-NF ==="
    )

    print(
        "Muestras época 0:",
        len(
            orden_epoch_0
        )
    )

    print(
        "Índices únicos época 0:",
        unicos_0
    )

    print(
        "Índices únicos época 1:",
        unicos_1
    )

    print(
        "Cobertura completa época 0:",
        cobertura_0
    )

    print(
        "Cobertura completa época 1:",
        cobertura_1
    )

    print(
        "Época 0 reproducible:",
        reproducible
    )

    print(
        "Épocas 0 y 1 tienen orden diferente:",
        orden_diferente
    )

    print(
        "Transiciones mismo archivo época 0:",
        round(
            pct_mismo_archivo,
            3
        ),
        "%"
    )


def main():

    validar_sampler()


if __name__ == "__main__":
    main()