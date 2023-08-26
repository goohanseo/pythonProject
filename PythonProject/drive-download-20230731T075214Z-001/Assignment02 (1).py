import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.models import resnet18
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import sys

print(sys.path)



class HelenDataSet(Dataset):
    def __init__(self, root_dir, partition, transform=None):
        self.root_dir = root_dir
        self.partition = partition
        self.transform = transform
        self.landmarks_frame = pd.read_csv(os.path.join(root_dir, f'{partition}names.txt'), sep=" ", header=None,
                                           names=["image", "num"])
        self.annotations_dir = os.path.join(root_dir, "annotation")

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.partition, self.landmarks_frame.iloc[idx, 0]) + ".jpg"
        image = Image.open(img_name).convert('RGB')
        img_first_line = os.path.splitext(self.landmarks_frame.iloc[idx, 0])[0]
        for file in os.listdir(self.annotations_dir):
            if file.endswith('.txt'):
                with open(os.path.join(self.annotations_dir, file), 'r') as f:
                    first_line = f.readline().strip()
                if first_line == img_first_line:
                    annotation_file = os.path.join(self.annotations_dir, file)
                    break
        landmarks = np.loadtxt(annotation_file, delimiter=',', dtype=np.float32, skiprows=1)
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample


class MyRescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        w, h = image.size
        new_h, new_w = self.output_size

        img = image.resize((new_h, new_w), resample=0)
        img = np.array(img)

        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class MyToTensor(object):
    def __init__(self):
        self.img_tensorfier = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = self.img_tensorfier(image)
        return {'image': image,
                'landmarks': torch.from_numpy(landmarks)}


root_dir = '/Users/guhanseo/Desktop/pythonProject/helen'

train_dataset = HelenDataSet(root_dir=root_dir, partition='train',
                             transform=transforms.Compose([MyRescale((224, 224)), MyToTensor()]))
test_dataset = HelenDataSet(root_dir=root_dir, partition='test',
                            transform=transforms.Compose([MyRescale((224, 224)), MyToTensor()]))
class show_landmarks(nn.Module):
    def __init__(self):
        super(show_landmarks, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=512, out_features=194 * 2, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        return x.view(-1, 194, 2)


batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                          collate_fn=torch.utils.data.dataloader.default_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = show_landmarks().to(device)

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
epoch_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        images, landmarks = batch['image'].to(device), batch['landmarks'].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.to(torch.float32), landmarks.to(torch.float32))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
with torch.no_grad():
    total_loss = 0
    for i, batch in enumerate(test_loader):
        images, landmarks = batch['image'].to(device), batch['landmarks'].to(device)

        outputs = model(images)
        loss = criterion(outputs, landmarks)

        total_loss += loss.item() * images.size(0)

    mean_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {mean_loss:.4f}')

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images, landmarks = batch['image'].to(device), batch['landmarks'].to(device)

        outputs = model(images)

        preds_np = outputs.cpu().numpy()[10]
        landmarks_np = landmarks.cpu().numpy()[10]

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        axs.imshow(images[10].cpu().numpy().transpose((1, 2, 0)))
        axs.scatter(preds_np[:, 0], preds_np[:, 1], s=10, marker='.', c='r')
        axs.scatter(landmarks_np[:, 0], landmarks_np[:, 1], s=10, marker='.', c='b')
        axs.axis('off')
        plt.show()

plt.plot(epoch_losses)
plt.title('Epoch Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()