"""Ejemplo comentado para analizar una melodía y generar visualizaciones."""

# --- Importación de librerías ---
from pathlib import Path  # Para manejar rutas de archivos al guardar imágenes.

# Se importan las herramientas necesarias para el análisis y la visualización.
import soundfile as sf  # Para leer archivos de audio.
import librosa  # Para rutinas de análisis de audio complementarias.
import librosa.display  # Para representar el espectrograma en un eje tiempo-frecuencia.
import matplotlib  # Para consultar qué backend se está usando.
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

    # Se consulta el backend activo de Matplotlib para decidir si se mostrarán ventanas.
    backend = matplotlib.get_backend()
    backend_lower = backend.lower()
    interactive_backend = not backend_lower.endswith("agg")

    if interactive_backend:
        print(f"El backend de Matplotlib es '{backend}', se mostrarán las figuras al final.")
    else:
        print(
            "Backend no interactivo (Agg u otro similar): las figuras se guardarán en disco.\n"
            "Para abrir ventanas interactivas puedes definir MPLBACKEND=TkAgg (u otro backend)\n"
            "antes de ejecutar el script, siempre que tengas las dependencias instaladas."
        )

    # --- Visualización del contorno melódico ---
    # Se asegura un directorio de salida para guardar las figuras generadas.
    output_dir = Path("salidas_visualizacion")
    output_dir.mkdir(exist_ok=True)

    # Se obtiene una figura con el contorno melódico usando el helper incluido en el paquete.
    contour_fig = plot_melody_contour(resultado)
    # Se define la ruta donde se guardará la imagen del contorno melódico.
    contour_path = output_dir / "contorno_melodico.png"
    # Se guarda la figura en disco con buena resolución.
    contour_fig.savefig(contour_path, dpi=150, bbox_inches="tight")
    # Si no se dispone de backend interactivo, se cierra la figura para liberar recursos.
    if not interactive_backend:
        plt.close(contour_fig)
    # Se informa en consola dónde quedó almacenada la imagen.
    print(f"Figura de contorno guardada en: {contour_path.resolve()}")

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

    # Se crea un gráfico del espectrograma en Matplotlib y se obtiene la figura resultante.
    manual_fig, ax = plt.subplots(figsize=(14, 4))
    # Se muestra el espectrograma en el gráfico, con ejes de tiempo y frecuencia en escala Mel.
    librosa.display.specshow(Sdb, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    # Se añade un título descriptivo al gráfico.
    ax.set_title("Mel-espectrograma (manual)")
    # Se ajusta el diseño para evitar solapamiento de elementos.
    manual_fig.tight_layout()
    # Se guarda la figura manual para contrastarla posteriormente con la versión anotada.
    manual_path = output_dir / "mel_espectrograma_manual.png"
    manual_fig.savefig(manual_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(manual_fig)
    # Se informa la ruta de la figura manual.
    print(f"Mel-espectrograma manual guardado en: {manual_path.resolve()}")

    # --- Visualización automatizada con secciones ---
    # Se genera una figura que reutiliza los segmentos detectados para resaltar cada bloque musical.
    sections_fig = plot_spectrogram_with_segments(y, sr, resultado)
    # Se guarda la figura con las secciones y etiquetas incrustadas.
    sections_path = output_dir / "mel_espectrograma_segmentado.png"
    sections_fig.savefig(sections_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(sections_fig)
    # Se muestra la ruta donde quedó guardado el espectrograma con secciones.
    print(f"Mel-espectrograma segmentado guardado en: {sections_path.resolve()}")

    # Si hay backend interactivo se realiza un plt.show() final para abrir las ventanas.
    if interactive_backend:
        print("Abriendo ventanas de Matplotlib...")
        plt.show()


if __name__ == "__main__":
    main()
