
# Limpieza y lematización de Orgullo y prejuicio

## Descripción
Este proyecto aplica técnicas de procesamiento de lenguaje natural
al libro *Orgullo y prejuicio*, de Jane Austen.

## Procesos realizados
- Lectura del archivo de texto.
- Normalización a minúsculas.
- Eliminación de signos de puntuación y palabras vacías.
- Lematización con spaCy.
- Exportación del texto procesado a `texto_limpio.txt`.

## Requisitos
Python y las dependencias indicadas en `requirements.txt`.

## Instalación
1. Crear y activar un entorno virtual.
2. Instalar las dependencias:

   `python -m pip install -r requirements.txt`

3. Instalar el modelo de español:

   `python -m spacy download es_core_news_sm`

## Ejecución
Ejecutar:

`python limpieza_texto.py`

## Archivos
- `limpieza_texto.py`: código de limpieza.
- `requirements.txt`: dependencias.
- `orgullo_y_prejuicio.txt`: texto original.
- `texto_limpio.txt`: resultado procesado.
