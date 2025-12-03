# MC-FSA

Herramienta experimental para segmentar y clasificar la estructura melódica de una grabación.
El flujo está inspirado en MSAF pero se centra en la melodía: extrae el contorno
(pitch y energía), detecta cambios estructurales y etiqueta cada frase con roles
musicales sencillos como "exposición", "pregunta" o "respuesta".

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Uso

Desde la línea de comandos:

```bash
python -m melody_analysis ruta/al/audio.wav \
    --output resultado.json \
    --melody-plot contorno.png \
    --sections-plot secciones.png
```

Los parámetros `--melody-plot` y `--sections-plot` guardan dos imágenes:
una con el contorno melódico extraído y otra con el espectrograma mel donde
se resaltan las secciones descritas en el JSON.

Si quieres experimentar con una copia independiente del pipeline sin tocar la
implementación original, hay un clon disponible bajo el nombre
`melody_analysis_v2` con los mismos puntos de entrada:

```bash
python -m melody_analysis_v2 ruta/al/audio.wav \
    --output resultado.json \
    --melody-plot contorno_v2.png \
    --sections-plot secciones_v2.png
```

En código:

```python
from melody_analysis import (
    MelodyAnalyzer,
    plot_melody_contour,
    plot_spectrogram_with_segments,
)
import librosa

analyzer = MelodyAnalyzer()
resultado = analyzer.analyze_file("ruta/al/audio.wav")
for segmento in resultado.segments:
    print(segmento.label, segmento.segment.start_time, segmento.segment.end_time)

# Generar visualizaciones directamente desde Python
fig1 = plot_melody_contour(resultado)
audio, sample_rate = librosa.load("ruta/al/audio.wav", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, resultado)
```

Si quieres usar directamente el clon con los colores mejorados (`melody_analysis_v2`)
desde un script como el que muestras (`from src.melody_analysis_v2 import ...`),
asegúrate primero de haber instalado el proyecto en editable (`pip install -e .`)
o de exportar `PYTHONPATH=src` antes de ejecutar el script. Luego importa sin el
prefijo `src` así:

```python
from melody_analysis_v2 import (
    MelodyAnalyzer,
    plot_melody_contour,
    plot_spectrogram_with_segments,
)
import librosa

analyzer = MelodyAnalyzer()
resultado = analyzer.analyze_file("1.mp3")
for segmento in resultado.segments:
    print(segmento.label, segmento.segment.start_time, segmento.segment.end_time)

fig1 = plot_melody_contour(resultado)
audio, sample_rate = librosa.load("1.mp3", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, resultado)
```

### Checklist rápido para usar el visualizador en tu propio script

Si ya tienes un script parecido al ejemplo anterior y parece que "no hace
nada", verifica estos puntos:

1) Instala el proyecto en editable (`pip install -e .[dev]`) o exporta
   `PYTHONPATH=src` en la misma sesión antes de ejecutarlo. Así las
   importaciones `from melody_analysis_v2 import ...` funcionarán sin el
   prefijo `src.`
2) Llama a las funciones de visualización (`plot_melody_contour` y
   `plot_spectrogram_with_segments`) igual que en el snippet y guarda o
   muestra las figuras:

```python
fig1 = plot_melody_contour(resultado)
fig1.savefig("contorno.png", dpi=150)
audio, sample_rate = librosa.load("1.mp3", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, resultado)
fig2.savefig("secciones.png", dpi=150)
```

3) Si quieres ver las ventanas interactivas, exporta un backend con soporte
   gráfico, por ejemplo `MPLBACKEND=TkAgg`, o ejecuta el script en un entorno
   que ya tenga backend interactivo. Si Matplotlib queda en modo `Agg`, las
   figuras se guardarán en disco (como en el ejemplo anterior) y no se
   abrirán ventanas.

Siguiendo esos pasos, tu snippet estará usando el visualizador tal como en el
clon `melody_analysis_v2` con colores para cada clasificación.

Y de forma análoga puedes importar `MelodyAnalyzer` desde
`melody_analysis_v2` para modificarlo libremente sin afectar al módulo
original.

### Ejemplo paso a paso con el clon `melody_analysis_v2`

Si prefieres un script listo para ejecutar que explique línea a línea el
flujo completo y genere las dos imágenes de soporte, revisa
`examples/visualizar_melodia_v2.py`. El código contiene comentarios en
español que describen cada instrucción, imprime por consola los segmentos
detectados y guarda en `salidas_visualizacion/` tanto el contorno melódico
como los dos espectrogramas (manual y segmentado).

El script intenta usar un backend interactivo (TkAgg/QtAgg/MacOSX) si hay
soporte gráfico disponible, de modo que también puedas ver las ventanas con
`plt.show()`. Si Matplotlib continúa utilizando `Agg`, exporta la variable
de entorno `MPLBACKEND` con el backend de tu preferencia (por ejemplo,
`MPLBACKEND=TkAgg`) antes de ejecutar el script y asegúrate de tener las
dependencias correspondientes instaladas.

```bash
python examples/visualizar_melodia_v2.py
```

Solo necesitas sustituir la ruta `1.mp3` que aparece en el script por tu
archivo de audio antes de ejecutarlo.

## Pruebas

```bash
pytest
```
