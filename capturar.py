import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pickle
import os


MODEL_PATH = 'modelo_faces.pth'
EMBEDDINGS_PATH = 'embeddings_promedio.pkl'
LABELS_PATH = 'labels.txt'
IMG_SIZE = 128
CAPTURAS_NECESARIAS = 20
DIR_CAPTURAS = 'capturas'


# Crear carpeta de respaldo si no existe
os.makedirs(DIR_CAPTURAS, exist_ok=True)

# Modelo CNN 
class CNNEmbeddingExtractor(nn.Module):
    def __init__(self):
        super(CNNEmbeddingExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.embedding(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNEmbeddingExtractor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def obtener_embedding(imagen):
    imagen = transform(imagen).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(imagen)
        return emb.cpu().numpy()[0]


if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = {}

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        label_names = [line.strip() for line in f.readlines()]
else:
    label_names = []


nombre = input("Nombre de la persona a registrar: ").strip()
if nombre in embeddings_db:
    print(f"Ya existe una persona con el nombre '{nombre}'.")
    exit()


ruta_persona = os.path.join(DIR_CAPTURAS, nombre)
os.makedirs(ruta_persona, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capturas = []

print("Presiona 'c' para capturar imagen cuando el rostro esté bien detectado. Presiona 'q' para salir.")

contador = 0
while len(capturas) < CAPTURAS_NECESARIAS:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        rostro = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Registro de nuevo rostro', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and len(faces) > 0:
        (x, y, w, h) = faces[0]
        rostro = frame[y:y+h, x:x+w]
        rostro_resized = cv2.resize(rostro, (IMG_SIZE, IMG_SIZE))
        capturas.append(rostro_resized)
        ruta_imagen = os.path.join(ruta_persona, f'{contador:03}.jpg')
        cv2.imwrite(ruta_imagen, rostro_resized)
        contador += 1
        print(f"[{len(capturas)}/{CAPTURAS_NECESARIAS}] Imagen capturada y guardada")

    if key == ord('q'):
        print("Cancelado por el usuario.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()


embeddings = [obtener_embedding(img) for img in capturas]
embedding_promedio = np.mean(embeddings, axis=0)

embeddings_db[nombre] = embedding_promedio
label_names.append(nombre)

with open(EMBEDDINGS_PATH, 'wb') as f:
    pickle.dump(embeddings_db, f)

with open(LABELS_PATH, 'w') as f:
    for name in label_names:
        f.write(name + '\n')

print(f"\n Persona '{nombre}' registrada con éxito y respaldo guardado en '{ruta_persona}'")
