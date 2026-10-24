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

# 读取数据
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([96, 96]),
        transforms.RandomRotation(45),
        transforms.CenterCrop(64),
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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
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

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
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

optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, filename='best.pt'):
    # model, dataloaders, criterion, optimizer, num_epochs=25,filename='best.pt'
    # resnet ，dataloaders，loss，Adam，epoch，模型存储文件
    # 咱们要算时间的
    since = time.time()
    # 记录acc最好的那一次
    best_acc = 0
    #     最后的epoch结果的准确率比中间epoch准确率的结果差也是有可能的
    # 模型也得放到你的CPU或者GPU
    model.to(device)
    # 训练过程中打印一堆损失和指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]
    #     optimizer.param_groups[0]是个字典结构，pytorch指定的
    # 最好的那次模型，后续会变的，先初始化
    best_model_wts = copy.deepcopy(model.state_dict())
    # 一个个epoch来遍历
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 放到你的CPU或GPU
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                outputs = model(inputs)
                #                 outputs：batch*102
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # 训练阶段更新权重
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 计算损失（先进行累加）
                running_loss += loss.item() * inputs.size(0)  # 0表示batch那个维度
                running_corrects += torch.sum(preds == labels.data)  # 预测结果最大的和真实值是否一致

            #             （除以数据集总数，得到平均的）
            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # 算平均
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since  # 一个epoch我浪费了多少时间
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 训练阶段和验证阶段都要进行前向传播，但是训练阶段还要进行参数更新，而验证阶段不需要
            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                # scheduler.step(epoch_loss)#学习率衰减
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()  # 学习率衰减

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)

    # 调用 plt.show() 之前保存图像
    plt.show()
    plt.savefig('./flower_result.png')

    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=5)

for param in model_ft.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=5)

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)
filename = 'best.pt'

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

dataiter = iter(dataloaders['valid'])
images, labels = next(dataiter)
model_ft.eval()
if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()
plt.savefig('./flower_result.png')



def test_model(model, dataloaders, device, cat_to_name, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0

    plt.figure(figsize=(20, 20))

    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 4, 4, images_so_far)
                ax.axis('off')
                ax.set_title(
                    f"predicted: {cat_to_name[str(preds[j].item())]} \n actual: {cat_to_name[str(labels[j].item())]}")
                plt.imshow(im_convert(inputs.cpu().data[j]))
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    plt.savefig('./flower_result.png')  # 在 plt.show() 之后保存图像
                    return
        model.train(mode=was_training)

    plt.show()
    plt.savefig('./flower_result.png')  # 在 plt.show() 之后保存图像


# 测试模型，展示预测结果
test_model(model_ft, dataloaders, device, cat_to_name)

# 保存最终模型
torch.save(model_ft.state_dict(), 'final_model.pth')

print("Training and evaluation complete. The model has been saved as 'final_model.pth'.")

# 额外功能：预测单张图片
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
