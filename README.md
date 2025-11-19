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
se resaltan las secciones descritas en el JSON. En la versión basada en
CREPE cada rol se colorea automáticamente para facilitar la lectura:

| Rol         | Color |
|-------------|-------|
| respuesta   | Azul  |
| pregunta    | Verde |
| exposición  | Morado|
| desarrollo  | Naranja |
| transición  | Rojo |
| cadencia    | Rosa |
| afirmación  | Café |

Si el clasificador produce una etiqueta diferente se dibuja en gris.

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

### ¿Cómo analizo mi propia canción?

1. **Consigue el archivo de audio** en tu equipo en un formato que `librosa`
   pueda leer (WAV, MP3, FLAC, etc.). No es necesario moverlo al repositorio;
   basta con conocer la ruta absoluta o relativa.
2. **Ejecuta la herramienta** apuntando a ese archivo. Por ejemplo, si tienes
   `mi_cancion.mp3` en la carpeta `~/musica/`, puedes invocar cualquiera de los
   módulos disponibles:

   ```bash
   python -m melody_analysis ~/musica/mi_cancion.mp3 \
       --output mi_cancion.json \
       --melody-plot mi_cancion_contorno.png \
       --sections-plot mi_cancion_secciones.png

   # Variante CREPE (instalando previamente el extra "crepe")
   python -m melody_analysis_v3 ~/musica/mi_cancion.mp3 \
       --output mi_cancion_crepe.json \
       --melody-plot mi_cancion_contorno_v3.png \
       --sections-plot mi_cancion_secciones_v3.png
   ```

3. **(Opcional) Convierte a WAV** si prefieres trabajar siempre con el mismo
   formato. Con `ffmpeg` sería:

   ```bash
   ffmpeg -i ~/musica/mi_cancion.mp3 ~/musica/mi_cancion.wav
   ```

4. **Modifica los ejemplos** para automatizar el proceso. En
   `examples/visualizar_melodia_v2.py` solo debes reemplazar la ruta `1.mp3`
   por tu archivo antes de ejecutar `python examples/visualizar_melodia_v2.py`;
   el script imprime los segmentos detectados y guarda las gráficas
   correspondientes en `salidas_visualizacion/`.

Además ahora existe una variante `melody_analysis_v3` que utiliza CREPE para
estimar el pitch. Este módulo replica el resto del pipeline pero permite
comparar la calidad de CREPE vs. la implementación basada en ``pyin``. Para
usar esta versión instala la dependencia opcional ``crepe``:

```bash
pip install .[crepe]
python -m melody_analysis_v3 ruta/al/audio.wav \
    --output resultado_crepe.json \
    --melody-plot contorno_v3.png \
    --sections-plot secciones_v3.png
```

Y de forma análoga puedes importar `MelodyAnalyzer` desde
`melody_analysis_v2` o `melody_analysis_v3` para modificarlo libremente sin
afectar al módulo original.

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
