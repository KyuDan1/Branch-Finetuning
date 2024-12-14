import os
import copy
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    # GradScaler 초기화
    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc = 0.0  # accuracy도 함께 추적한다고 가정

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 옵티마 그래디언트 초기화
                optimizer.zero_grad()

                # **훈련 단계**에서 autocast와 scaler 적용
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        # Mixed Precision 구간
                        with autocast():  # float16 연산 사용
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                        # 역전파: scaler 사용
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    else:
                        # 검증 단계
                        with torch.no_grad():
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 베스트 모델 저장
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc



def branch_finetune(model, dataloaders, criterion, optimizer, epochs_per_branch, num_branches):
    best_model = None
    best_loss = float('inf')
    best_acc = 0.0

    for branch in range(num_branches):
        print(f'Branch {branch + 1}/{num_branches}')
        model, val_loss, val_acc = train_model(model, dataloaders, criterion, optimizer, epochs_per_branch)

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            best_model = copy.deepcopy(model)

    print(f'Best Validation Loss: {best_loss:.4f}, Best Validation Accuracy: {best_acc:.4f}')
    return best_model, best_loss, best_acc


def linear_finetune(model, dataloaders, criterion, optimizer, num_epochs):
    print("Starting linear fine-tuning")
    model, val_loss, val_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs)
    print(f'Linear Fine-Tuning Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    return model, val_loss, val_acc


# Example use case
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset and transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


from torchvision.transforms.functional import to_tensor

def collate_fn_with_tensor_conversion(batch):
    # Convert PIL images in the batch to tensors
    images, labels = zip(*batch)
    images = [to_tensor(img) for img in images]
    return torch.stack(images), torch.tensor(labels)


# Load dataset (CIFAR-10 as an example)
dataset = datasets.CIFAR10(root='.data', train=True, download=True, transform=data_transform)
data_train, data_val = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(data_train, batch_size=2, shuffle=True)
val_loader = DataLoader(data_val, batch_size=2, shuffle=False)


dataloaders = {
    'train': train_loader,
    'val': val_loader
}

# Define model (ResNet18)
print("model loading")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes
model.to(device)
print("model loading complete")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Branch finetuning")
# Train using the branching fine-tuning method
final_branch_model, branch_loss, branch_acc = branch_finetune(copy.deepcopy(model), dataloaders, criterion, optimizer, epochs_per_branch=5, num_branches=3)
print("linear finetuning")
# Train using the linear fine-tuning method
final_linear_model, linear_loss, linear_acc = linear_finetune(copy.deepcopy(model), dataloaders, criterion, optimizer, num_epochs=15)

# Compare results
print(f"Branch Fine-Tuning -> Loss: {branch_loss:.4f}, Accuracy: {branch_acc:.4f}")
print(f"Linear Fine-Tuning -> Loss: {linear_loss:.4f}, Accuracy: {linear_acc:.4f}")
