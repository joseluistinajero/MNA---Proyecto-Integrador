# Preparación y manifiesto de datasets

## Objetivo

Este directorio documenta el proceso utilizado para identificar, registrar y preparar los fragmentos de video que posteriormente serán utilizados para entrenar y evaluar los diferentes modelos del proyecto.

La unidad de análisis definida actualmente es un **clip de 4 segundos**. Cada clip se registra en un archivo CSV denominado `dataset_manifest.csv`, que funciona como índice maestro de los fragmentos identificados en los diferentes datasets.

El manifiesto permite mantener trazabilidad entre cada clip, su video de origen, la etiqueta asignada y el evento al que pertenece.

---

## Datasets considerados

Actualmente se consideran los siguientes datasets:

- **DCSASS / Shoplifting**
- **MNNIT**
- **CCTV_YOLO**

GitHub se utiliza para mantener el código, la documentación y los archivos necesarios para reproducir el proceso de preparación de datos.

---

## Archivo `dataset_manifest.csv`

El manifiesto utiliza actualmente las siguientes columnas:

```text
dataset,video_id,video_origen,clip_id,inicio_s,fin_s,etiqueta,persona_id,evento_id,nota
```

### Descripción de las columnas

| Columna | Descripción |
|---|---|
| `dataset` | Dataset del cual proviene el video, por ejemplo `DCSASS`, `MNNIT` o `CCTV_YOLO`. |
| `video_id` | Identificador único asignado al video de origen. |
| `video_origen` | Nombre del archivo de video original. |
| `clip_id` | Identificador único del clip de 4 segundos. |
| `inicio_s` | Segundo de inicio del clip dentro del video original. |
| `fin_s` | Segundo final del clip dentro del video original. |
| `etiqueta` | Clase asignada al clip. Actualmente: `ocultamiento`, `sin_ocultamiento` o `no_determinado`. |
| `persona_id` | Identificador de la persona relevante dentro del video. |
| `evento_id` | Identificador del evento de ocultamiento. Permite relacionar varios clips que pertenecen al mismo evento. |
| `nota` | Observaciones adicionales sobre el clip o las condiciones de la escena. |

---

## Reglas actuales de identificación de clips

### Duración

La duración objetivo de cada unidad de análisis es de **4 segundos**.

Los tiempos del manifiesto se expresan en segundos.

### Etiquetado

Se utilizan actualmente tres etiquetas:

- `ocultamiento`: dentro del intervalo existe evidencia visible de un evento de ocultamiento.
- `sin_ocultamiento`: no existe un evento de ocultamiento dentro del intervalo.
- `no_determinado`: la evidencia visual no permite decidir con suficiente claridad si ocurrió o no un ocultamiento.

Si un evento de ocultamiento aparece parcialmente dentro de un clip, el clip se etiqueta como `ocultamiento`.

Si un evento abarca más de un intervalo de 4 segundos, todos los clips que contengan parte visible del evento pueden etiquetarse como `ocultamiento`.

---

## Uso de `evento_id`

Cuando varios clips corresponden al mismo evento físico de ocultamiento, deben compartir el mismo `evento_id`.

Ejemplo:

```text
clip_id: MNNIT_001_C001
inicio_s: 12
fin_s: 16
evento_id: E001

clip_id: MNNIT_001_C002
inicio_s: 16
fin_s: 20
evento_id: E001
```

Esto permite identificar que ambos clips corresponden al mismo evento aunque sean unidades de análisis diferentes.

---

## Uso de `persona_id`

`persona_id` identifica a la persona relevante dentro de cada video.

Los identificadores son locales al video y pueden expresarse, por ejemplo, como:

```text
P01
P02
P03
```

No se pretende identificar a una persona entre videos diferentes.

---

## Particularidades por dataset

### DCSASS / Shoplifting

En este dataset los archivos parecen estar previamente segmentados en clips de aproximadamente 4 segundos.

Para estos casos:

```text
inicio_s = 0
fin_s = duración del clip
```

Aunque el archivo ya corresponda directamente a la unidad de análisis, se le asignará un `clip_id` para mantener el mismo esquema utilizado en los demás datasets.

Ejemplo:

```csv
DCSASS,DCSASS_001,Shoplifting001.mp4,DCSASS_001_C001,0,4,ocultamiento,P01,E001,"Clip ya segmentado en el dataset"
```

---

### MNNIT

Los videos no se encuentran previamente divididos en clips de 4 segundos.

El manifiesto registrará el intervalo temporal identificado en el video original.

Ejemplo:

```csv
MNNIT,MNNIT_001,video001.mp4,MNNIT_001_C001,12,16,ocultamiento,P01,E001,"Ocultamiento visible durante parte del intervalo"
```

Un `clip_id` puede existir inicialmente como identificador lógico aunque todavía no se haya generado físicamente el archivo de video correspondiente.

Posteriormente, un script podrá utilizar `video_origen`, `inicio_s` y `fin_s` para generar el clip.

---

### CCTV_YOLO

Al igual que MNNIT, los videos se analizarán utilizando intervalos temporales registrados en el manifiesto.

Ejemplo:

```csv
CCTV_YOLO,CCTV_001,video001.mp4,CCTV_001_C001,24,28,ocultamiento,P01,E001,"Persona oculta producto"
```

Si la acción no puede determinarse claramente, puede registrarse como:

```csv
CCTV_YOLO,CCTV_002,video002.mp4,CCTV_002_C001,40,44,no_determinado,P02,,"Acción parcialmente ocluida"
```

---

## Regla para las particiones de entrenamiento

Los clips derivados de un mismo `video_id` no deben distribuirse entre diferentes particiones del dataset.

Por ejemplo, no debe ocurrir:

```text
MNNIT_001_C001 -> train
MNNIT_001_C002 -> test
```

Si ambos clips provienen de:

```text
video_id = MNNIT_001
```

entonces todos los clips derivados de ese video deben permanecer en la misma partición:

```text
MNNIT_001 -> train
```

Esta regla busca evitar fuga de información entre los conjuntos de entrenamiento, validación y prueba.

---

## Flujo de trabajo

El proceso previsto es:

```text
Videos originales
        ↓
Revisión visual
        ↓
Identificación de intervalos
        ↓
dataset_manifest.csv
        ↓
Validación del manifiesto
        ↓
Generación de clips
        ↓
Preparación para entrenamiento
        ↓
Train / Validation / Test
```

El manifiesto se construye antes de generar físicamente todos los clips, de manera que la selección y etiquetado de los intervalos quede documentada y pueda reproducirse posteriormente.

---

## Ejemplo de `dataset_manifest.csv`

```csv
dataset,video_id,video_origen,clip_id,inicio_s,fin_s,etiqueta,persona_id,evento_id,nota
DCSASS,DCSASS_001,Shoplifting001.mp4,DCSASS_001_C001,0,4,ocultamiento,P01,E001,"Clip ya segmentado en el dataset"
DCSASS,DCSASS_002,Shoplifting002.mp4,DCSASS_002_C001,0,4,sin_ocultamiento,P01,,"Clip ya segmentado en el dataset"
MNNIT,MNNIT_001,video001.mp4,MNNIT_001_C001,12,16,ocultamiento,P01,E001,"Ocultamiento visible durante parte del intervalo"
MNNIT,MNNIT_001,video001.mp4,MNNIT_001_C002,16,20,ocultamiento,P01,E001,"Continuación del mismo evento"
MNNIT,MNNIT_002,video002.mp4,MNNIT_002_C001,8,12,sin_ocultamiento,P01,,"Comportamiento normal"
CCTV_YOLO,CCTV_001,video001.mp4,CCTV_001_C001,24,28,ocultamiento,P01,E001,"Persona oculta producto"
CCTV_YOLO,CCTV_001,video001.mp4,CCTV_001_C002,28,32,sin_ocultamiento,P01,,"Sin ocultamiento en esta ventana"
CCTV_YOLO,CCTV_002,video002.mp4,CCTV_002_C001,40,44,no_determinado,P02,,"Acción parcialmente ocluida"
```

---

## Almacenamiento de los videos

Los archivos de video no se almacenan en este repositorio.

El esquema actual de trabajo es:

```text
Servidor BNext
    almacenamiento principal de datasets

Google Drive
    copia de trabajo para revisión colaborativa

GitHub
    código
    documentación
    scripts
    manifiesto o ejemplos de manifiesto
```

Esta separación permite mantener el repositorio ligero y enfocado en los elementos necesarios para reproducir el proceso de preparación de datos.