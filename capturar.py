import cv2
import os


dataset_path = "Celebrity Faces Dataset"

nombre_persona = input("Ingresa el nombre de la persona: ").strip().replace(" ", "_")
ruta_persona = os.path.join(dataset_path, nombre_persona)


os.makedirs(ruta_persona, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

contador = 0
max_imagenes = 50  

print(f"\n[INFO] Capturando imágenes para: {nombre_persona}")
print("[INFO] Presiona 'q' para salir antes o 'Espacio' para capturar manualmente.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo acceder a la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    caras = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in caras:
        rostro = frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (224, 224))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Captura automática cada 5 frames si se detecta rostro
        if contador < max_imagenes:
            img_path = os.path.join(ruta_persona, f"{nombre_persona}_{contador:03d}.jpg")
            cv2.imwrite(img_path, rostro)
            contador += 1
            print(f"[CAPTURADO] Imagen {contador}/{max_imagenes}")

        if contador >= max_imagenes:
            print("[COMPLETADO] Captura finalizada.")
            break

    cv2.imshow("Captura de rostro", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or contador >= max_imagenes:
        break

cap.release()
cv2.destroyAllWindows()
