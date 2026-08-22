import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os

class WaterDataset(Dataset):
    def __init__(self, root_dir, train=True, size=256):
        self.files = []
        for label in [0,1]:
            folder = os.path.join(root_dir, str(label))
            for f in os.listdir(folder):
                self.files.append((os.path.join(folder,f), label))

        if train:
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(90), 
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), #256 -> 128
            nn.Conv2d(16, 32, 3, padding=1),nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), #128->64
            nn.Conv2d(32, 64, 3, padding=1),nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2) #64->32 
        )
        self.fc = nn.Sequential (
            nn.Flatten(),
            nn.Linear(64*32*32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2) #бинарная классификация вода \ суша
        )

    def forward(self, x):
        return self.fc(self.conv(x))


if __name__ == "__main__":
    torch.set_num_threads(8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на: {device}")

    train_dataset = WaterDataset("train", train=True)
    val_dataset = WaterDataset("val", train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    epochs = 20  
    best_val_acc = 0.0  
    
    print("Начинаем обучение...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Эпоха [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"  Валидация Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "water_classifier_best.pth")
            print(f"  ** Новый лучший результат: {best_val_acc:.2f}% (сохранено в water_classifier_best.pth) **")

    print("Обучение завершено.")
    print(f"Итоговая лучшая точность на валидации: {best_val_acc:.2f}%")