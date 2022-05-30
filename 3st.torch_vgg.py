import torch.optim as optim
import json
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from torchvision.datasets import ImageFolder

random.seed(777)
torch.manual_seed(777)

batch_size = 10
learning_rate = 0.0002
training_epochs = 15



cfgs = {
"A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
"E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

with open("./imagenet1000_clsidx_to_labels.json", "r") as read_file:
    class_idx = json.load(read_file)
idx2label = [class_idx[str(k)] for k in range(len(class_idx))]
cls2label = {class_idx[str(k)][0]:class_idx[str(k)][1] for k in range(len(class_idx))}

train_path = '../dataset/imagenet/train'
train_imgs = ImageFolder(train_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_imgs,  batch_size = batch_size,shuffle = True)

class VGG(torch.nn.Module):
    def __init__(self, num_classes= 1000, init_weights = True):
        super().__init__()        
        self.conv = nn.Sequential(
                    #3 224 128
                    nn.Conv2d(3, 64, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.MaxPool2d(2, 2),
                    #64 112 64
                    nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.MaxPool2d(2, 2),
                    #128 56 32
                    nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.MaxPool2d(2, 2),
                    #256 28 16
                    nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.MaxPool2d(2, 2),
                    #512 14 8
                    nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                    nn.MaxPool2d(2, 2)
        )
        
        self.avgpool = nn.AvgPool2d(7)
        
        self.classifier = nn.Linear(512, 1000)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
            features = self.conv(x)
            x = self.avgpool(features)
            x = x.view(features.size(0), -1)
            x = self.classifier(x)
            return x, features
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def make_layers(cfg, batch_norm = False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layes += [nn.MaxPool2d(kernel_size = 2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

device = "cpu"
print(f'Target Device {device}')
vgg_net = VGG().to(device)

param = list(vgg_net.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_net.parameters(), lr = 0.0001)


# In[35]:


for epoch in range(training_epochs):
    running_loss = 0.0
    if epoch > 0:
        vgg_net = VGG()
        vgg_net.load_state_dict(torch.load(save_path))
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)#copy data to system memory
        optimizer.zero_grad()
        outputs, f = vgg_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if loss.item() > 1000:
            print(loss.item())
            for param in vgg_net.parameters():
                print(param.data)
        running_loss +=  loss.item()
        if i % 50 == 49:
            print('[%d, 5%d] loss: %.3f' % (epoch+1, i+1, running_loss / 50))
            running_loss = 0.0
    save_path = 'vgg_image_net.pth'
    torch.save(vgg_net.state_dict(), save_path)
print('Finish training')


# In[21]:


device = "cpu"
model_path = 'vgg_image_net.pth'

vgg_net = VGG().to(device)
vgg_net.load_state_dict(torch.load(model_path))


# In[22]:


class_correct = list(0. for i in range(1000))
class_total = list(0. for i in range(1000))


# In[37]:


val_path = '../dataset/imagenet/val/'
val_imgs = ImageFolder(val_path, transform=transform)
val_loader = torch.utils.data.DataLoader(val_imgs,batch_size = batch_size,shuffle = True)


# In[42]:


testloader = val_loader
print(f'Batch size {batch_size}')
with torch.no_grad():
    
#     for data in testloader:
    for batch_idx, data in enumerate(testloader):
        full_size = len(testloader.dataset)
        print((batch_idx / full_size) * 100)
        images, labels = data
        images = images.cpu()
        labels = labels.cpu()

        outputs,_ = vgg_net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_sum=0
for i in range(1000):
    temp = 100 * class_correct[i] / class_total[i]
    print('Accuracy of %5s : %2d %%' % (idx2label[i], temp))
    accuracy_sum+=temp
print('Accuracy average: ', accuracy_sum/1000)


# In[ ]:




