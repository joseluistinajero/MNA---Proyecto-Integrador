# MNA---Proyecto-Integrador
Repositorio principal de trabajo y evidencia académica, del Proyecto Integrador "Detección de eventos de ocultamiento de mercancía mediante Inteligencia Artificial" de la Maestría en Inteligencia Artificial Aplicada del Tecnológico de Monterrey, manteniendo una estructura organizada y accesible para los integrantes del equipo y el asesor.

# 🤖 Detección de Eventos de Ocultamiento de Mercancía mediante Inteligencia Artificial

¡Bienvenidos al repositorio de nuestro proyecto integrador! Este espacio está destinado a la organización, almacenamiento y colaboración de los entregables académicos y técnicos desarrollados para la **Maestría en Inteligencia Artificial Aplicada** del **Instituto Tecnológico de Estudios Superiores de Monterrey**.

El proyecto se realiza en colaboración con la empresa **BLOCK NETWORKS S.A. DE C.V. (BNext)** dentro del departamento de Desarrollo (Monterrey, Nuevo León).

---

## 👥 Integrantes del Equipo
* **Jesús Alberto Jiménez Ramos** - Matrícula: `A01796903`
* **Nélida Silva Cabrera** - Matrícula: `A01796696`
* **José Luis Tinajero Guerrero** - Matrícula: `A01797178`

### 🏢 Stakeholders
* **Sponsor / Patrocinador:** Manuel Jiménez (Director de Desarrollo, BNext)
* **Correo institucional del Sponsor:** mjimenez@bnext.mx

---

## 🚀 Descripción del Proyecto
* **Título Oficial:** Evaluación de modelos de visión computacional para la detección de eventos de ocultamiento de mercancía en video.
* **Dominio Principal:** Visión Computacional & Reconocimiento de Acciones en Video.

### Resumen Técnico
El objetivo primordial de este proyecto consiste en extraer información relevante directamente del contenido visual de secuencias de video (provenientes de cámaras de vigilancia en entornos de comercio minorista) para **identificar patrones visuales y temporales asociados con acciones de ocultamiento de mercancía**. 

A través del análisis evolutivo de los movimientos a lo largo del tiempo, evaluaremos y compararemos dos grandes enfoques metodológicos:
1. **Modelos basados en secuencias RGB directas de video** (utilizando arquitecturas como *R3D-18* como baseline inicial).
2. **Modelos basados en estimación y secuencias de pose**, representando el movimiento del cuerpo mediante puntos corporales clave para reducir la dependencia de variables del entorno (ropa, iluminación, etc.).

Finalmente, el proyecto entregará una recomendación técnica sustentada y una evaluación documental para determinar la factibilidad de desplegar el modelo óptimo en **hardware Edge**.

* **Palabras clave:** visión computacional, reconocimiento de acciones, análisis de video, ocultamiento de mercancía, detección de acciones, videovigilancia, comercio minorista.

---

## 📁 Estructura del Repositorio

Siguiendo la estructura inicial planificada, los recursos se organizan en los siguientes directorios:

* 📁 **`Documentacion/`**: Documentos relacionados con el desarrollo, seguimiento y los entregables académicos del proyecto.
* 📁 **`Notebooks/`**: Cuadernos de Jupyter (`.ipynb`) utilizados para el análisis exploratorio, la experimentación, el entrenamiento y la evaluación de modelos.
* 📁 **`Datos/`**: Información sobre los datasets utilizados, su origen, estructura, condiciones de acceso y restricciones de uso acordadas.
* 📄 **`README.md`**: Descripción del proyecto, objetivos, integrantes y organización general.

> 🔒 **Nota sobre confidencialidad (NDA):** En cumplimiento con el acuerdo de confidencialidad con BNext, los conjuntos de datos masivos o sensibles y códigos internos restringidos se respaldarán de manera privada en el servidor provisto por la empresa, evitando incorporar credenciales o información confidencial a este repositorio.

---

## 📋 Plan y Seguimiento de Entregables

Marca con una `x` (ejemplo: `[x]`) las tareas completadas a lo largo de las semanas del bloque:

- [ ] **Semana 1:** Planteamiento del proyecto
- [ ] **Semana 2:** Avance 0. Propuesta de proyecto y firma de convenios
- [ ] **Semana 3:** Avance 1. Caracterización y análisis exploratorio del conjunto de videos (EDA)
- [ ] **Semana 4:** Avance 2. Preparación y representación espacio-temporal de los videos
- [ ] **Semana 5:** Avance 3. Desarrollo y evaluación del modelo *Baseline*
- [ ] **Semana 6:** Avance 4. Evaluación y comparación de enfoques/modelos alternativos
- [ ] **Semana 7:** Avance 5. Selección del modelo final y evaluación de factibilidad en hardware *Edge*
- [ ] **Semana 8:** Avance 6. Producto de difusión
- [ ] **Semana 9:** Avance 7. Resumen ejecutivo
- [ ] **Semanas 10-11:** Presentación final del Proyecto Integrador

---

## 🛠️ Tecnologías y Frameworks Tentativos
* **Lenguaje base:** Python 3.x
* **Modelos & Video:** PyTorch / torchvision (arquitectura R3D-18 sugerida), OpenCV
* **Estimación de Pose:** MediaPipe / OpenPose / YOLOv8-Pose *(según se determine en la semana 4)*
* **Análisis de datos:** Pandas, NumPy, Matplotlib, Seaborn

## Práctica Git

Esta sección fue agregada para practicar el control de versiones con Git.

Esta segunda línea todavía no ha sido agregada al staging area.
