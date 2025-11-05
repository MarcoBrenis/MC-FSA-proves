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
