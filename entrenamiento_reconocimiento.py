import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import pickle
import random


dataset_path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset")
dataPath = os.path.join(dataset_path, 'Celebrity Faces Dataset')
print("Path to dataset files:", dataset_path)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

images, labels = [], []
label_names, label_map = [], {}
current_label = 0

for person in os.listdir(dataPath):
    person_path = os.path.join(dataPath, person)
    if not os.path.isdir(person_path):
        continue
    files = os.listdir(person_path)
    if len(files) < 30:
        print(f"[IGNORADO] {person} tiene menos de 30 imágenes ({len(files)})")
        continue
    print(f"[CARGANDO] {person}: {len(files)} imágenes encontradas.")
    label_map[person] = current_label
    label_names.append(person)
    for file in files:
        img_path = os.path.join(person_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(current_label)
    current_label += 1

if len(images) == 0:
    raise RuntimeError("No se encontraron imágenes suficientes.")

X = torch.stack(images)
y = torch.tensor(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Modelo
class ResNetEmbeddingExtractor(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        for name, param in self.resnet.named_parameters():
            param.requires_grad = False
        for name, param in self.resnet.layer4.named_parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.embedding(x)

embedding_size = 128
extractor = ResNetEmbeddingExtractor(embedding_size)

class ClasificadorDesdeEmbeddings(nn.Module):
    def __init__(self, extractor, num_classes):
        super().__init__()
        self.extractor = extractor
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.extractor(x)
        return self.fc(x)

model = ClasificadorDesdeEmbeddings(extractor, len(label_names))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Entrenamiento
train_losses = []
best_accuracy = 0.0
best_model_state = None

for epoch in range(30):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss)
    print(f"Época {epoch+1}, Pérdida: {total_loss:.4f}")

    # Evaluación
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = 100 * correct / total
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model.state_dict()

print(f"\nMejor precisión alcanzada: {best_accuracy:.2f}%")
model.load_state_dict(best_model_state)
torch.save(model.extractor.state_dict(), 'modelo_faces.pth')

with open('labels.txt', 'w') as f:
    for name in label_names:
        f.write(name + '\n')

# Embeddings promedio
persona_embeddings = {}

def obtener_embedding(modelo, imagen_tensor):
    modelo.eval()
    with torch.no_grad():
        return modelo(imagen_tensor.unsqueeze(0)).squeeze().cpu().numpy()

for label_id in range(len(label_names)):
    indices = [i for i, y_val in enumerate(y) if y_val == label_id]
    imgs = X[indices]
    embs = [obtener_embedding(model.extractor, img) for img in imgs]
    persona_embeddings[label_names[label_id]] = np.mean(embs, axis=0)

with open("embeddings_promedio.pkl", "wb") as f:
    pickle.dump(persona_embeddings, f)

# Curva de pérdida
plt.figure()
plt.plot(train_losses, marker='o')
plt.title("Curva de pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.grid(True)
plt.savefig("curva_perdida_entrenamiento.png")
plt.show()

# Matriz de confusión
model.eval()
predictions, targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        targets.extend(y_batch.numpy())

cm = confusion_matrix(targets, predictions)
plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay(cm, display_labels=label_names).plot(xticks_rotation=90)
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.show()

# Visualización real vs predicción
print("Mostrando imágenes reales vs predichas...")
fig, axs = plt.subplots(3, 6, figsize=(18, 9))
axs = axs.flatten()
sample_indices = random.sample(range(len(X_test)), min(18, len(X_test)))
for i, idx in enumerate(sample_indices):
    img = X_test[idx].permute(1, 2, 0).detach().numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    axs[i].imshow(img)
    axs[i].axis('off')
    real_name = label_names[y_test[idx]]
    pred_name = label_names[predictions[idx]]
    axs[i].set_title(f"Real: {real_name}\nPred: {pred_name}", fontsize=9)
plt.tight_layout()
plt.savefig("comparacion_real_vs_predicho.png")
plt.show()
