import numpy as np


ORDEN_COCO18_STGNF = np.array(
    [
        0,
        17,
        6,
        8,
        10,
        5,
        7,
        9,
        12,
        14,
        16,
        11,
        13,
        15,
        2,
        1,
        4,
        3
    ],
    dtype=np.int64
)


NOMBRES_COCO18_STGNF = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear"
]


def limpiar_pose_retails(pose_coco17):

    pose = np.asarray(
        pose_coco17,
        dtype=np.float32
    ).copy()

    if pose.shape != (
        17,
        3
    ):
        raise ValueError(
            "La pose RetailS debe tener shape (17, 3)."
        )

    if not np.isfinite(
        pose[
            :,
            :2
        ]
    ).all():
        raise ValueError(
            "Se encontraron coordenadas x o y no finitas."
        )

    confianza = pose[
        :,
        2
    ]

    confianza = np.where(
        np.isfinite(
            confianza
        ),
        confianza,
        0.0
    )

    confianza = np.clip(
        confianza,
        0.0,
        1.0
    )

    pose[
        :,
        2
    ] = confianza

    return pose


def coco17_a_coco18_stgnf(pose_coco17):

    pose_coco17 = np.asarray(
        pose_coco17,
        dtype=np.float32
    )

    if pose_coco17.shape[-2:] != (
        17,
        3
    ):
        raise ValueError(
            "La pose debe terminar en dimensiones (17, 3)."
        )

    cuello = 0.5 * (
        pose_coco17[
            ...,
            5,
            :
        ]
        + pose_coco17[
            ...,
            6,
            :
        ]
    )

    pose_con_cuello = np.concatenate(
        [
            pose_coco17,
            cuello[
                ...,
                None,
                :
            ]
        ],
        axis=-2
    )

    return pose_con_cuello[
        ...,
        ORDEN_COCO18_STGNF,
        :
    ]

def obtener_tramos_consecutivos(frame_ids):
    frames = np.asarray(
        sorted(
            set(
                int(frame_id)
                for frame_id in frame_ids
            )
        ),
        dtype=np.int64
    )

    if frames.size == 0:
        return []

    cortes = np.where(
        np.diff(frames) != 1
    )[0]

    limites = np.concatenate(
        [
            np.array([0]),
            cortes + 1,
            np.array([len(frames)])
        ]
    )

    return [
        frames[
            limites[i]:
            limites[i + 1]
        ]
        for i in range(
            len(limites) - 1
        )
    ]

def validar_preparacion_retails_stgnf():

    assert len(ORDEN_COCO18_STGNF) == 18
    assert len(NOMBRES_COCO18_STGNF) == 18

    tramos = obtener_tramos_consecutivos(
        [
            0,
            1,
            2,
            5,
            6,
            10
        ]
    )

    tramos_obtenidos = [
        tramo.tolist()
        for tramo in tramos
    ]

    tramos_esperados = [
        [0, 1, 2],
        [5, 6],
        [10]
    ]

    assert tramos_obtenidos == tramos_esperados

    pose_prueba = np.zeros(
        (
            1,
            2,
            2,
            3
        ),
        dtype=np.float32
    )

    pose_prueba[
        0,
        :,
        :,
        0
    ] = [
        [100.0, 200.0],
        [300.0, 400.0]
    ]

    pose_prueba[
        0,
        :,
        :,
        1
    ] = [
        [100.0, 200.0],
        [300.0, 400.0]
    ]

    normalizada = normalizar_pose_stgnf(
        pose_prueba,
        np.array(
            [
                1080.0,
                1920.0
            ],
            dtype=np.float32
        )
    )

    assert normalizada.shape == (
        1,
        2,
        2,
        3
    )

    assert np.isfinite(
        normalizada
    ).all()

    print("=== VALIDACIÓN PREPARACIÓN RETAILS PARA STG-NF ===")
    print("Articulaciones COCO18:", len(ORDEN_COCO18_STGNF))
    print("Nombres COCO18:", len(NOMBRES_COCO18_STGNF))
    print("Separación de tramos consecutivos:", tramos_obtenidos)
    print("Validación básica: OK")


def normalizar_pose_stgnf(
    pose_ntvc,
    vid_res
):

    pose = np.asarray(
        pose_ntvc,
        dtype=np.float32
    ).copy()

    if pose.ndim != 4:
        raise ValueError(
            "La entrada debe tener formato N x T x V x C."
        )

    if pose.shape[-1] != 3:
        raise ValueError(
            "La última dimensión debe contener "
            "x, y y confianza."
        )

    norm_factor = np.array(
        [
            float(vid_res[0]),
            float(vid_res[1]),
            1.0
        ],
        dtype=np.float32
    )

    pose_normalizada = (
        pose
        / norm_factor
    )

    media_xy = pose_normalizada[
        ...,
        :2
    ].mean(
        axis=(
            1,
            2
        )
    )

    std_y = pose_normalizada[
        ...,
        1
    ].std(
        axis=(
            1,
            2
        )
    )

    if np.any(
        std_y <= 0
    ):
        raise ValueError(
            "Se encontró una secuencia con desviación "
            "estándar vertical igual a cero."
        )

    pose_normalizada[
        ...,
        :2
    ] = (
        pose_normalizada[
            ...,
            :2
        ]
        - media_xy[
            :,
            None,
            None,
            :
        ]
    ) / std_y[
        :,
        None,
        None,
        None
    ]

    return pose_normalizada

def main():

    validar_preparacion_retails_stgnf()


if __name__ == "__main__":
    main()