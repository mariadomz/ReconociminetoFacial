# Sistema de Reconocimiento Facial y Emocional

## 📸 Entrenamiento del Modelo

Ejecuta `entrenamiento.py` para:

- Descargar el dataset de celebridades desde KaggleHub.
- Filtrar personas con al menos **30 imágenes**.
- Entrenar un modelo **ResNet18** para extraer embeddings.
- Guardar los **embeddings promedio por persona**.
- Visualizar:
  - **Curva de pérdida**
  - **Matriz de confusión**
  - **Predicciones reales vs estimadas**

### Comando

```bash
python entrenamiento.py
```

---

## 👤 Reconocimiento en Vivo + Emociones

Ejecuta `reconocimiento.py` para:

- Capturar imagen desde una **cámara IP** (o adaptarlo para webcam).
- Detectar caras con **Haar Cascade**.
- Comparar el embedding extraído con la base de embeddings promedio usando **similitud coseno**.
- Detectar emociones usando **FER** (`mtcnn=True`).
- Mostrar:
  - Persona reconocida
  - Nivel de similitud
  - Emoción detectada

### Comando

```bash
python reconocimiento.py
```

> ⚠️ **Nota**: Asegúrate de modificar el enlace de la cámara (`VideoCapture`) si no estás usando la misma IP.

---

## 📷 Captura de Nuevas Imágenes

El script `captura.py` permite capturar nuevas imágenes de una persona para futuras sesiones de entrenamiento. Las capturas se guardan en una carpeta local organizada por nombre.

### Comando

```bash
python captura.py
```

---

## 📊 Resultados del Modelo

Los siguientes archivos se generan automáticamente luego del entrenamiento:

- **Curva de pérdida:** `curva_perdida_entrenamiento.png`
- **Matriz de confusión:** `matriz_confusion.png`
- **Comparación real vs predicción:** `comparacion_real_vs_predicho.png`

---

> ✨ Asegúrate de revisar los gráficos generados para evaluar el desempeño del modelo.
