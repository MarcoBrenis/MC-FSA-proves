Herramientas experimentales para segmentar y clasificar melodías inspiradas en
MSAF (Music Structure Analysis Framework).

## Instalación

Se recomienda utilizar un entorno virtual con Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Ejemplo de ejecución para segmentar un archivo de audio y obtener la
clasificación retórica de cada segmento.

```bash
python segment_melody.py ruta/al/audio.wav --segments 6 --json analisis.json
```

La extracción de la melodía puede hacerse tanto con `librosa.pyin` (por
defecto) como con [CREPE](https://github.com/marl/crepe). Para comparar ambos
enfoques basta con indicar el backend deseado y, opcionalmente, el tamaño de
paso (en milisegundos) utilizado por CREPE:

```bash
python segment_melody.py ruta/al/audio.wav --pitch-backend crepe --crepe-step-ms 10
```

El uso de CREPE requiere instalar su dependencia (incluida en
`requirements.txt`).

El comando anterior generará un archivo JSON con la información de cada
segmento: tiempo de inicio y fin, etiqueta (por ejemplo, "Pregunta",
"Respuesta", "Exposición del tema"), valores descriptivos de pendiente
melódica, rango, energía, etc.

## Desarrollo

La lógica principal se encuentra en `melody_analysis/segmenter.py`, donde se:

1. Extrae un contorno melódico con `librosa.pyin`.
2. Construye una matriz de similitud basada en características melódicas y de
   energía.
3. Calcula una curva de novedad y detecta posibles fronteras estructurales.
4. Clasifica cada segmento con reglas heurísticas inspiradas en el análisis
   musical (pregunta-respuesta, exposición del tema, clímax, desarrollo).

El código está modularizado para facilitar ajustes en las reglas de
clasificación o en la detección de límites.
