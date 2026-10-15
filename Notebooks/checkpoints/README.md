# Artefactos del baseline R3D-18

Este directorio contiene los artefactos asociados al baseline R3D-18 evaluado
en el Avance 3 del Proyecto Integrador.

## Checkpoint del modelo

El checkpoint entrenado se mantiene fuera de Git debido a su tamaño.

Archivo requerido:

`r3d18_baseline_best.pt`

Tamaño aproximado:

`235 MB`

SHA-256:

`08ad02ad8d10153187343c6a1b781a2a3cd2b686b216f8945c8a20b2b9d477bd`

Para reproducir exactamente el baseline documentado en el notebook, el archivo
debe colocarse en:

`Notebooks/checkpoints/r3d18_baseline_best.pt`

El notebook verifica automáticamente la integridad del checkpoint mediante
SHA-256 antes de utilizarlo.

El equipo conserva copias externas verificadas del checkpoint para respaldo y
recuperación.

## Artefactos versionados

Los siguientes archivos se conservan en el repositorio:

- `r3d18_baseline_historial.csv`
- `r3d18_baseline_test_predictions.csv`
- `r3d18_baseline_experiment_manifest.json`

La partición experimental utilizada se conserva en:

`Datos/Preparacion_Datasets/baseline_split_equipo41.csv`
