"""
Utilidades para controlar la ejecución del ajuste de
hiperparámetros de los modelos RGB del Avance 4.
"""

import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import gc
import pandas as pd

def decidir_accion_tuning(
    estado_artefactos,
    ejecutar_tuning=False,
    force_retrain=False
):
    """
    Determina qué hacer con un experimento de tuning.

    Estados esperados de artefactos:
    - "validos"
    - "ausentes"
    - "incompatibles"

    Retorna:
    - "REUTILIZAR"
    - "ENTRENAR"
    - "REENTRENAR"
    - "OMITIR"
    - "BLOQUEAR"
    """

    estados_validos = {
        "validos",
        "ausentes",
        "incompatibles"
    }

    if estado_artefactos not in estados_validos:
        raise ValueError(
            "Estado de artefactos no reconocido: "
            f"{estado_artefactos}"
        )

    # --------------------------------------------------------
    # Artefactos incompatibles
    # --------------------------------------------------------

    if estado_artefactos == "incompatibles":
        return "BLOQUEAR"

    # --------------------------------------------------------
    # Artefactos válidos
    # --------------------------------------------------------

    if estado_artefactos == "validos":

        if (
            ejecutar_tuning
            and force_retrain
        ):
            return "REENTRENAR"

        return "REUTILIZAR"

    # --------------------------------------------------------
    # Artefactos ausentes
    # --------------------------------------------------------

    if ejecutar_tuning:
        return "ENTRENAR"

    return "OMITIR"


def validar_control_tuning():
    """
    Autocomprobación de la lógica de protección.
    """

    casos = [
        (
            "ausentes",
            False,
            False,
            "OMITIR"
        ),
        (
            "ausentes",
            True,
            False,
            "ENTRENAR"
        ),
        (
            "validos",
            False,
            False,
            "REUTILIZAR"
        ),
        (
            "validos",
            True,
            False,
            "REUTILIZAR"
        ),
        (
            "validos",
            True,
            True,
            "REENTRENAR"
        ),
        (
            "incompatibles",
            False,
            False,
            "BLOQUEAR"
        ),
        (
            "incompatibles",
            True,
            False,
            "BLOQUEAR"
        )
    ]

    for (
        estado,
        ejecutar,
        force,
        esperado
    ) in casos:

        obtenido = decidir_accion_tuning(
            estado_artefactos=estado,
            ejecutar_tuning=ejecutar,
            force_retrain=force
        )

        if obtenido != esperado:
            raise RuntimeError(
                "Fallo en control de tuning: "
                f"{estado=}, "
                f"{ejecutar=}, "
                f"{force=}, "
                f"{obtenido=}, "
                f"{esperado=}"
            )

    return True

from pathlib import Path


SLUG_MODELOS_TUNING = {
    "R(2+1)D-18": "r2plus1d_18",
    "MC3-18": "mc3_18"
}


def construir_rutas_tuning(
    checkpoint_dir,
    modelo,
    configuracion
):
    """
    Construye rutas reproducibles para checkpoint e historial
    de una configuración de tuning.
    """

    if modelo not in SLUG_MODELOS_TUNING:
        raise ValueError(
            f"Modelo no soportado para tuning: {modelo}"
        )

    slug = SLUG_MODELOS_TUNING[modelo]
    config_slug = configuracion.lower()

    checkpoint_path = (
        Path(checkpoint_dir)
        / f"{slug}_avance4_{config_slug}.pt"
    )

    historial_path = (
        Path(checkpoint_dir)
        / f"{slug}_avance4_{config_slug}_history.csv"
    )

    return checkpoint_path, historial_path


def construir_configuracion_tuning(
    modelo,
    configuracion,
    hiperparametros,
    seed,
    batch_size,
    max_epochs,
    dataset_sha256
):
    """
    Construye la identidad experimental que se guardará
    dentro del checkpoint.
    """

    return {
        "schema_version": 1,
        "fase": "tuning",
        "modelo": modelo,
        "configuracion": configuracion,
        "seed": seed,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr_representacion": (
            hiperparametros["lr_representacion"]
        ),
        "lr_clasificador": (
            hiperparametros["lr_clasificador"]
        ),
        "weight_decay": (
            hiperparametros["weight_decay"]
        ),
        "dataset_v2_sha256": dataset_sha256,
        "num_frames": 16,
        "metrica_checkpoint": "average_precision"
    }

def crear_optimizador_tuning_rgb(
    modelo,
    hiperparametros
    ):
    """
    Construye el optimizador AdamW para MC3-18 y
    R(2+1)D-18 conservando dos grupos de learning rate.
    """

    return optim.AdamW(
        [
            {
                "params":
                    modelo.layer4[1].parameters(),
                "lr":
                    hiperparametros[
                        "lr_representacion"
                    ],
                "nombre":
                    "representacion"
            },
            {
                "params":
                    modelo.fc.parameters(),
                "lr":
                    hiperparametros[
                        "lr_clasificador"
                    ],
                "nombre":
                    "clasificador"
            }
        ],
        weight_decay=(
            hiperparametros[
                "weight_decay"
            ]
        )
    )

def construir_plan_tuning_rgb(
    modelos,
    espacio_tuning,
    config_batch,
    checkpoint_dir,
    repo_dir,
    seed,
    max_epochs,
    dataset_sha256,
    ejecutar_tuning,
    force_retrain,
    experimentos_habilitados,
    validar_artefactos_fn
):
    """
    Construye el plan protegido de experimentos de tuning RGB.

    H0 se excluye porque corresponde al baseline ya ejecutado.
    Esta función no entrena modelos ni modifica artefactos.
    """

    plan = []

    for modelo in modelos:

        batch_size = (
            config_batch[modelo]["batch_size"]
        )

        for nombre_config, hiperparametros in (
            espacio_tuning.items()
        ):

            if nombre_config == "H0":
                continue

            checkpoint_path, historial_path = (
                construir_rutas_tuning(
                    checkpoint_dir=checkpoint_dir,
                    modelo=modelo,
                    configuracion=nombre_config
                )
            )

            config_experimento = (
                construir_configuracion_tuning(
                    modelo=modelo,
                    configuracion=nombre_config,
                    hiperparametros=hiperparametros,
                    seed=seed,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    dataset_sha256=dataset_sha256
                )
            )

            validacion = validar_artefactos_fn(
                checkpoint_path=checkpoint_path,
                historial_path=historial_path,
                configuracion_esperada=config_experimento
            )

            clave_experimento = (
                modelo,
                nombre_config
            )

            ejecutar_experimento = (
                ejecutar_tuning
                and clave_experimento
                in experimentos_habilitados
            )
            accion = decidir_accion_tuning(
                estado_artefactos=validacion["estado"],
                ejecutar_tuning=ejecutar_experimento,
                force_retrain=force_retrain
            )
            plan.append(
                {
                    "modelo": modelo,
                    "configuracion": nombre_config,
                    "lr_representacion": (
                        hiperparametros[
                            "lr_representacion"
                        ]
                    ),
                    "lr_clasificador": (
                        hiperparametros[
                            "lr_clasificador"
                        ]
                    ),
                    "weight_decay": (
                        hiperparametros[
                            "weight_decay"
                        ]
                    ),
                    "estado_artefactos": (
                        validacion["estado"]
                    ),
                    "accion": accion,
                    "checkpoint": str(
                        checkpoint_path.relative_to(
                            repo_dir
                        )
                    )
                }
            )

    return plan

def preparar_reproducibilidad_tuning(
    seed,
    loader_train
    ):
    """
    Reinicia las fuentes de aleatoriedad antes de cada
    experimento de tuning para mantener condiciones
    comparables entre configuraciones.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    sampler = getattr(
        loader_train,
        "sampler",
        None
    )

    generator = getattr(
        sampler,
        "generator",
        None
    )

    if generator is not None:
        generator.manual_seed(seed)

def ejecutar_experimento_tuning_rgb(
    modelo,
    configuracion,
    hiperparametros,
    accion,
    seed,
    device,
    usar_amp,
    max_epochs,
    checkpoint_path,
    historial_path,
    config_experimento,
    loader_train,
    loader_val,
    crear_modelo_fn,
    configurar_parametros_fn,
    entrenar_experimento_fn
):
    """
    Ejecuta un único experimento de tuning RGB únicamente
    cuando la acción autorizada es ENTRENAR o REENTRENAR.

    OMITIR y REUTILIZAR no inicializan modelos ni ejecutan
    entrenamiento.
    """

    acciones_validas = {
        "OMITIR",
        "REUTILIZAR",
        "ENTRENAR",
        "REENTRENAR",
        "BLOQUEAR"
    }

    if accion not in acciones_validas:
        raise ValueError(
            f"Acción de tuning no reconocida: {accion}"
        )

    # --------------------------------------------------------
    # Estados que no deben producir entrenamiento
    # --------------------------------------------------------

    if accion in {
        "OMITIR",
        "REUTILIZAR"
    }:
        return {
            "modelo": modelo,
            "configuracion": configuracion,
            "accion": accion,
            "entrenado": False,
            "checkpoint_path": checkpoint_path,
            "historial_path": historial_path,
            "resultado": None
        }

    if accion == "BLOQUEAR":
        raise RuntimeError(
            "Experimento bloqueado por artefactos "
            f"incompatibles: {modelo} / {configuracion}"
        )

    # --------------------------------------------------------
    # Protección contra planes obsoletos
    # --------------------------------------------------------

    if accion == "ENTRENAR":

        if (
            checkpoint_path.exists()
            or historial_path.exists()
        ):
            raise RuntimeError(
                "ENTRENAR bloqueado porque ya existen "
                "artefactos para "
                f"{modelo} / {configuracion}. "
                "Reconstruye el plan antes de continuar."
            )

    # --------------------------------------------------------
    # Reproducibilidad
    # --------------------------------------------------------

    preparar_reproducibilidad_tuning(
        seed=seed,
        loader_train=loader_train
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    modelo_tuning = crear_modelo_fn(
        modelo,
        num_classes=2,
        pretrained=True
    )

    modelo_tuning = configurar_parametros_fn(
        modelo_tuning,
        modelo
    )

    modelo_tuning = modelo_tuning.to(
        device
    )

    # --------------------------------------------------------
    # Optimizador
    # --------------------------------------------------------

    optimizador = crear_optimizador_tuning_rgb(
        modelo=modelo_tuning,
        hiperparametros=hiperparametros
    )

    # --------------------------------------------------------
    # Pérdida y precisión mixta
    # --------------------------------------------------------

    criterio = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            usar_amp
            and device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # Entrenamiento común RGB
    # --------------------------------------------------------

    resultado = entrenar_experimento_fn(
        modelo=modelo_tuning,
        nombre_modelo=modelo,
        loader_train=loader_train,
        loader_val=loader_val,
        criterio=criterio,
        optimizador=optimizador,
        scaler=scaler,
        device=device,
        max_epochs=max_epochs,
        checkpoint_path=checkpoint_path,
        historial_path=historial_path,
        experiment_config=config_experimento,
        usar_amp=usar_amp
    )

    return {
        "modelo": modelo,
        "configuracion": configuracion,
        "accion": accion,
        "entrenado": True,
        "checkpoint_path": checkpoint_path,
        "historial_path": historial_path,
        "resultado": resultado
    }
def validar_barrera_ejecucion_tuning():
    """
    Comprueba que OMITIR impide cualquier inicialización
    o entrenamiento del modelo.
    """

    def no_debe_ejecutarse(*args, **kwargs):
        raise RuntimeError(
            "Se alcanzó código de entrenamiento "
            "cuando la acción era OMITIR."
        )

    resultado = ejecutar_experimento_tuning_rgb(
        modelo="R(2+1)D-18",
        configuracion="H1",
        hiperparametros={},
        accion="OMITIR",
        seed=42,
        device=None,
        usar_amp=True,
        max_epochs=15,
        checkpoint_path=None,
        historial_path=None,
        config_experimento={},
        loader_train=None,
        loader_val=None,
        crear_modelo_fn=no_debe_ejecutarse,
        configurar_parametros_fn=no_debe_ejecutarse,
        entrenar_experimento_fn=no_debe_ejecutarse
    )

    if resultado["entrenado"]:
        raise RuntimeError(
            "OMITIR produjo un experimento marcado "
            "como entrenado."
        )

    if resultado["accion"] != "OMITIR":
        raise RuntimeError(
            "La acción devuelta no coincide con OMITIR."
        )

    return True

def ejecutar_tuning_rgb(
    plan,
    espacio_tuning,
    config_batch,
    checkpoint_dir,
    seed,
    device,
    usar_amp,
    max_epochs,
    dataset_sha256,
    loaders_rgb,
    crear_modelo_fn,
    configurar_parametros_fn,
    entrenar_experimento_fn
):
    """
    Ejecuta secuencialmente el plan protegido de tuning RGB.

    El plan ya contiene la acción autorizada para cada
    experimento. H0 no debe formar parte de esta fase.
    """

    resultados = []

    for experimento in plan:

        modelo = experimento["modelo"]
        configuracion = experimento["configuracion"]
        accion = experimento["accion"]

        if configuracion == "H0":
            raise RuntimeError(
                "H0 no debe ejecutarse durante el tuning."
            )

        hiperparametros = (
            espacio_tuning[configuracion]
        )

        batch_size = (
            config_batch[modelo]["batch_size"]
        )

        checkpoint_path, historial_path = (
            construir_rutas_tuning(
                checkpoint_dir=checkpoint_dir,
                modelo=modelo,
                configuracion=configuracion
            )
        )

        config_experimento = (
            construir_configuracion_tuning(
                modelo=modelo,
                configuracion=configuracion,
                hiperparametros=hiperparametros,
                seed=seed,
                batch_size=batch_size,
                max_epochs=max_epochs,
                dataset_sha256=dataset_sha256
            )
        )

        resultado = ejecutar_experimento_tuning_rgb(
            modelo=modelo,
            configuracion=configuracion,
            hiperparametros=hiperparametros,
            accion=accion,
            seed=seed,
            device=device,
            usar_amp=usar_amp,
            max_epochs=max_epochs,
            checkpoint_path=checkpoint_path,
            historial_path=historial_path,
            config_experimento=config_experimento,
            loader_train=(
                loaders_rgb[modelo]["train"]
            ),
            loader_val=(
                loaders_rgb[modelo]["val"]
            ),
            crear_modelo_fn=crear_modelo_fn,
            configurar_parametros_fn=(
                configurar_parametros_fn
            ),
            entrenar_experimento_fn=(
                entrenar_experimento_fn
            )
        )

        resumen = None

        if resultado["resultado"] is not None:
            resumen = resultado["resultado"].get(
                "resumen"
            )

        resultados.append(
            {
                "modelo": resultado["modelo"],
                "configuracion": resultado["configuracion"],
                "accion": resultado["accion"],
                "entrenado": resultado["entrenado"],
                "checkpoint_path": (
                    resultado["checkpoint_path"]
                ),
                "historial_path": (
                    resultado["historial_path"]
                ),
                "resumen": resumen
            }
        )

        del resultado

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return resultados

def main():

    validar_control_tuning()
    validar_barrera_ejecucion_tuning()

    print(
        "Control de tuning RGB validado correctamente."
    )

    print(
        "Barrera de ejecución OMITIR validada correctamente."
    )

def resumir_resultados_tuning(
    comparacion_rgb,
    checkpoint_dir,
    modelos,
    configuraciones
):
    """
    Construye una tabla comparable entre el baseline H0
    y las configuraciones H1-H4 usando la mejor época
    según Average Precision de validación.
    """

    filas = []

    for modelo in modelos:

        # ----------------------------------------------------
        # H0: baseline ya ejecutado
        # ----------------------------------------------------

        base = (
            comparacion_rgb.loc[
                comparacion_rgb["modelo"] == modelo
            ]
            .iloc[0]
        )

        filas.append(
            {
                "modelo": modelo,
                "configuracion": "H0",
                "mejor_epoca": int(
                    base["mejor_epoca"]
                ),
                "val_ap": float(
                    base["val_average_precision"]
                ),
                "val_pr_auc": float(
                    base["val_pr_auc"]
                ),
                "val_roc_auc": float(
                    base["val_roc_auc"]
                ),
                "val_eer": float(
                    base["val_eer"]
                ),
                "val_f1": float(
                    base["val_f1_05"]
                )
            }
        )

        # ----------------------------------------------------
        # H1-H4
        # ----------------------------------------------------

        for configuracion in configuraciones:

            _, historial_path = (
                construir_rutas_tuning(
                    checkpoint_dir=checkpoint_dir,
                    modelo=modelo,
                    configuracion=configuracion
                )
            )

            if not historial_path.exists():
                raise FileNotFoundError(
                    "Falta el historial de "
                    f"{modelo} {configuracion}"
                )

            historial = pd.read_csv(
                historial_path
            )

            idx_mejor = (
                historial[
                    "val_average_precision"
                ]
                .idxmax()
            )

            mejor = historial.loc[
                idx_mejor
            ]

            filas.append(
                {
                    "modelo": modelo,
                    "configuracion": configuracion,
                    "mejor_epoca": int(
                        mejor["epoch"]
                    ),
                    "val_ap": float(
                        mejor[
                            "val_average_precision"
                        ]
                    ),
                    "val_pr_auc": float(
                        mejor["val_pr_auc"]
                    ),
                    "val_roc_auc": float(
                        mejor["val_roc_auc"]
                    ),
                    "val_eer": float(
                        mejor["val_eer"]
                    ),
                    "val_f1": float(
                        mejor["val_f1"]
                    )
                }
            )

    return (
        pd.DataFrame(filas)
        .sort_values(
            [
                "modelo",
                "val_ap"
            ],
            ascending=[
                True,
                False
            ]
        )
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    main()