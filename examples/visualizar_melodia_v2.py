"""Ejemplo comentado para analizar una melodía y generar visualizaciones."""

# --- Importación de librerías ---
# Se importan las herramientas necesarias para el análisis y la visualización.
import soundfile as sf  # Para leer archivos de audio.
import librosa  # Para rutinas de análisis de audio complementarias.
import librosa.display  # Para representar el espectrograma en un eje tiempo-frecuencia.
import matplotlib.pyplot as plt  # Para crear y mostrar gráficos.
import numpy as np  # Para operaciones numéricas, especialmente con arrays.

# Se importa la clase principal y los auxiliares de visualización del clon v2.
from melody_analysis_v2 import (
    MelodyAnalyzer,  # Encapsula la extracción, segmentación y clasificación.
    plot_melody_contour,  # Función para graficar el contorno melódico.
    plot_spectrogram_with_segments,  # Función para graficar el espectrograma con secciones.
)


def main() -> None:
    """Ejecuta el análisis sobre un archivo y muestra los resultados."""

    # --- Análisis de la Melodía ---
    # Se define la ruta al archivo de audio que se quiere analizar.
    audio_path = "1.mp3"
    # Se crea una instancia del analizador de melodías.
    analyzer = MelodyAnalyzer()
    # Se llama al método para analizar el archivo, que devuelve un objeto con los resultados.
    resultado = analyzer.analyze_file(audio_path)

    # Se imprime en la consola un resumen de los segmentos detectados.
    print("Segmentos detectados:")
    # Se recorre cada segmento en los resultados para mostrar sus datos principales.
    for segmento in resultado.segments:
        # Se imprime la etiqueta, la hora de inicio y la de final con tres decimales.
        print(f"{segmento.label:>12} | {segmento.segment.start_time:7.3f} → {segmento.segment.end_time:7.3f} s")

    # --- Visualización del contorno melódico ---
    # Se obtiene una figura con el contorno melódico usando el helper incluido en el paquete.
    contour_fig = plot_melody_contour(resultado)
    # Se muestra la figura en pantalla para inspeccionar los segmentos y etiquetas.
    contour_fig.show()

    # --- Visualización manual del Mel-espectrograma ---
    # Se carga el archivo de audio con soundfile para acceder a la señal (y) y la frecuencia de muestreo (sr).
    y, sr = sf.read(audio_path)
    # Si el audio es estéreo (más de un canal), se convierte a mono promediando los canales.
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Se calcula el Mel-espectrograma; representa la energía por bandas perceptuales a lo largo del tiempo.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # Se convierte la potencia a decibelios para resaltar detalles finos.
    Sdb = librosa.power_to_db(S, ref=np.max)

    # Se crea un gráfico del espectrograma en Matplotlib.
    plt.figure(figsize=(14, 4))
    # Se muestra el espectrograma en el gráfico, con ejes de tiempo y frecuencia en escala Mel.
    librosa.display.specshow(Sdb, sr=sr, x_axis="time", y_axis="mel")
    # Se añade un título descriptivo al gráfico.
    plt.title("Mel-espectrograma (manual)")
    # Se ajusta el diseño para evitar solapamiento de elementos.
    plt.tight_layout()
    # Se muestra el gráfico manual para contrastarlo con la versión anotada.
    plt.show()

    # --- Visualización automatizada con secciones ---
    # Se genera una figura que reutiliza los segmentos detectados para resaltar cada bloque musical.
    sections_fig = plot_spectrogram_with_segments(y, sr, resultado)
    # Se muestra la figura con las secciones y etiquetas incrustadas.
    sections_fig.show()


if __name__ == "__main__":
    main()
