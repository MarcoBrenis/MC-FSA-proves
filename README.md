diff --git a/README.md b/README.md
index 64a1eca4ce32a02c0465c77d21203ffd0e95332a..d033b7c5cae30c0d190dcb192ae37ace85f69e45 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,42 @@
-# MC-FSA
+# MC-FSA
+
+Herramientas experimentales para segmentar y clasificar melodías inspiradas en
+MSAF (Music Structure Analysis Framework).
+
+## Instalación
+
+Se recomienda utilizar un entorno virtual con Python 3.10+.
+
+```bash
+python -m venv .venv
+source .venv/bin/activate
+pip install -r requirements.txt
+```
+
+## Uso
+
+Ejemplo de ejecución para segmentar un archivo de audio y obtener la
+clasificación retórica de cada segmento.
+
+```bash
+python segment_melody.py ruta/al/audio.wav --segments 6 --json analisis.json
+```
+
+El comando anterior generará un archivo JSON con la información de cada
+segmento: tiempo de inicio y fin, etiqueta (por ejemplo, "Pregunta",
+"Respuesta", "Exposición del tema"), valores descriptivos de pendiente
+melódica, rango, energía, etc.
+
+## Desarrollo
+
+La lógica principal se encuentra en `melody_analysis/segmenter.py`, donde se:
+
+1. Extrae un contorno melódico con `librosa.pyin`.
+2. Construye una matriz de similitud basada en características melódicas y de
+   energía.
+3. Calcula una curva de novedad y detecta posibles fronteras estructurales.
+4. Clasifica cada segmento con reglas heurísticas inspiradas en el análisis
+   musical (pregunta-respuesta, exposición del tema, clímax, desarrollo).
+
+El código está modularizado para facilitar ajustes en las reglas de
+clasificación o en la detección de límites.

