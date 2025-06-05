📸 Entrenamiento del Modelo
Ejecuta entrenamiento.py para:

Descargar el dataset de celebridades desde KaggleHub.

Filtrar personas con al menos 30 imágenes.

Entrenar un modelo ResNet18 para extraer embeddings.

Guardar los embeddings promedio por persona.

Visualizar curva de pérdida, matriz de confusión y predicciones reales vs estimadas.

bash
Copiar
Editar
python entrenamiento.py
👤 Reconocimiento en Vivo + Emociones
Ejecuta reconocimiento.py para:

Capturar la imagen desde una cámara IP (o se puede adaptar a webcam).

Detectar caras con Haar Cascade.

Comparar el embedding extraído con la base de embeddings promedio usando similitud coseno.

Detectar emociones usando FER (mtcnn=True).

Mostrar la persona reconocida, la similitud y la emoción detectada.

bash
Copiar
Editar
python reconocimiento.py
Asegúrate de modificar el enlace de la cámara (VideoCapture) si no estás usando la misma IP.

📷 Captura de Nuevas Imágenes
El script captura.py permite capturar nuevas imágenes para futuras personas a agregar. Guarda capturas en una carpeta local para luego ser usadas en entrenamiento.

bash
Copiar
Editar
python captura.py
📊 Resultados del Modelo
Curva de pérdida: curva_perdida_entrenamiento.png

Matriz de confusión: matriz_confusion.png

Comparación real vs predicción: comparacion_real_vs_predicho.png

Estos archivos se generan automáticamente después del entrenamiento.