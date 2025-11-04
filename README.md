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
python -m melody_analysis ruta/al/audio.wav --output resultado.json
```

Si quieres experimentar con una copia independiente del pipeline sin tocar la
implementación original, hay un clon disponible bajo el nombre
`melody_analysis_v2` con los mismos puntos de entrada:

```bash
python -m melody_analysis_v2 ruta/al/audio.wav --output resultado.json
```

En código:

```python
from melody_analysis import MelodyAnalyzer

analyzer = MelodyAnalyzer()
resultado = analyzer.analyze_file("ruta/al/audio.wav")
for segmento in resultado.segments:
    print(segmento.label, segmento.segment.start_time, segmento.segment.end_time)
```

Y de forma análoga puedes importar `MelodyAnalyzer` desde
`melody_analysis_v2` para modificarlo libremente sin afectar al módulo
original.

## Pruebas

```bash
pytest
```
