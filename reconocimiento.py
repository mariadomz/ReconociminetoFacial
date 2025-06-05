import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from fer import FER  
import time

MODEL_PATH = 'modelo_faces.pth'
EMBEDDINGS_PATH = 'embeddings_promedio.pkl'
LABELS_PATH = 'labels.txt'
THRESHOLD = 0.7  
IMG_SIZE = 224   


with open(LABELS_PATH, 'r') as f:
    label_names = [line.strip() for line in f.readlines()]


with open(EMBEDDINGS_PATH, 'rb') as f:
    embeddings_db = pickle.load(f)

# Modelo ResNet 
class ResNetEmbeddingExtractor(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        from torchvision.models import resnet18
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.embedding(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetEmbeddingExtractor(embedding_size=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def obtener_embedding(imagen):
    imagen = transform(imagen).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(imagen)
        return emb.cpu().numpy()[0]

#  detector de emociones FER
detector_emociones = FER(mtcnn=True)  


cap = cv2.VideoCapture("http://192.168.0.2:8080/video")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        rostro = frame[y:y+h, x:x+w]
        rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)

        
        embedding = obtener_embedding(rostro_rgb)

        
        mejor_similitud = -1
        nombre_predicho = "Desconocido"

        for nombre, emb_db in embeddings_db.items():
            similitud = cosine_similarity([embedding], [emb_db])[0][0]
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                if similitud >= THRESHOLD:
                    nombre_predicho = nombre
                else:
                    nombre_predicho = "Desconocido"

        # Detectar emoción con FER (pasando solo imagen RGB)
        try:
            emociones = detector_emociones.detect_emotions(rostro_rgb)
            if emociones:
                emocion_dominante = max(emociones[0]["emotions"], key=emociones[0]["emotions"].get)
                prob_emocion = emociones[0]["emotions"][emocion_dominante]
                texto_emocion = f"{emocion_dominante} ({prob_emocion:.2f})"
            else:
                texto_emocion = "Emocion: N/A"
        except Exception:
            texto_emocion = "Emocion: Error"

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{nombre_predicho} ({mejor_similitud:.2f})", (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, texto_emocion, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Reconocimiento Facial y Emociones', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
