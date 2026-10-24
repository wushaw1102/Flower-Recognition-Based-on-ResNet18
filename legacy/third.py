import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import time
import warnings

warnings.filterwarnings("ignore")
import random
import copy
import json
from PIL import Image

# 修改: 添加早停和混合精度训练
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# 修改: 更新数据增强策略
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(64),  # 新增
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 128
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model_name = 'resnet'
feature_extract = True
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# 修改: 使用更深的ResNet模型
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 64
    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

filename = 'best.pt'

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

optimizer_ft = optim.Adam(params_to_update, lr=1e-3)
# 修改: 使用ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()


# 修改: 更新训练函数，添加早停和混合精度训练
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='best.pt', patience=10):
    since = time.time()
    best_acc = 0
    model.to(device)
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    # 早停
    early_stopping_counter = 0
    early_stopping_best_acc = 0

    # 混合精度训练
    scaler = GradScaler()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid':
                scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    state = {
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state, filename)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print("Early stopping triggered")
                    model.load_state_dict(best_model_wts)
                    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


# 训练模型
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,
                                                                                            num_epochs=25, patience=10)

# 绘制训练结果
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(train_acc_history, label='train')
plt.plot(val_acc_history, label='val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./flower_result.png')
plt.show()

# 加载最佳模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])


# 测试模型函数保持不变
def im_convert(tensor):
    """ 将张量转换为可显示的图像 """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def test_model(model, dataloaders, device, cat_to_name, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(20, 20))

    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(num_images // 4, 4, images_so_far)
                ax.axis('off')
                ax.set_title(
                    f"predicted: {cat_to_name[str(preds[j].item())]} \n actual: {cat_to_name[str(labels[j].item())]}")

                # 使用 im_convert 函数
                img = im_convert(inputs.cpu()[j])
                plt.imshow(img)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.savefig('./flower_result.png')
                    plt.show()
                    return

    model.train(mode=was_training)
    plt.tight_layout()
    plt.savefig('./flower_result.png')
    plt.show()

# 测试模型
test_model(model_ft, dataloaders, device, cat_to_name)

# 保存最终模型
torch.save(model_ft.state_dict(), 'final_model.pth')

print("Training and evaluation complete. The model has been saved as 'final_model.pth'.")


# 预测单张图片的函数保持不变
def predict_image(image_path, model, device, cat_to_name, topk=5):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_labels = torch.topk(probabilities, topk)
        top_probs = top_probs.cpu().numpy()
        top_labels = top_labels.cpu().numpy()

    class_names = [cat_to_name[str(label)] for label in top_labels]
    return top_probs, class_names

# 示例：预测单张图片
image_path = './flower_data/valid/1/image_06739.jpg'
probs, classes = predict_image(image_path, model_ft, device, cat_to_name)

print("Top probabilities:", probs)
print("Predicted classes:", classes)