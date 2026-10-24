import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
import imageio
import time
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image
import torch
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt


#读取数据
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    'train':
        transforms.Compose([
#     Compose：按顺序进行组合
        transforms.Resize([96, 96]),
#             不管原数据的大小，规定用于训练的图片的大小，Resize根据实际来
       #以下6行代码是数据增强的过程，数据不够时，通过数据增强，更高效的利用数据，平移，翻转，放大等方法让数据具有更多的多样性
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(64),#从中心开始裁剪
#            从96*96 随机裁剪64*64也有无数种可能性
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
#             p=0.5指的是每张图像有50%的裁剪可能性
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
#         #参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相，不是重点，考虑极端光线条件
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
#            转成pytorch专用格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
#             标准化 参数来源是大数据集的参数，由于有三个颜色通道，R,G,B，所以有三个μ和σ
#             标准化(x-μ)/σ
    ]),
    'valid':
#     验证集不需要再进行图像加强的过程，用原有的数据进行验证即可
        transforms.Compose([
        transforms.Resize([64, 64]),
#             数据大小要和训练集一样大
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         验证集和测试集标准化采用的均值，标准差要一致
    ]),
}

batch_size = 128
# batchsize比较大，是因为图片是64*64的比较小
# 通过文件夹来获取数据和标签
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# for x in ['train', 'valid']文件夹名字
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
#  shuffle=True 表示在每个迭代中是否对数据进行打乱，
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
# class_names顺序是1,10,100,101,102，先预测1开头的，再预测2开头的

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#     json文件，每个分类的名字
# 加载models中提供的模型，并且直接用训练的好权重当做初始化参数
# 第一次执行需要下载，可能会比较慢
model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
feature_extract = True #都用人家特征，先不更新
# 特征提取，用人家的方法，把所有层都冻住，只保留输出层
# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
#     model：resnet
#     feature_extracting：true
    if feature_extracting:
        for param in model.parameters():
#             遍历模型中的每一个参数
            param.requires_grad = False
#     反向更新的参数设置为False，，不计算梯度，参数就不更新

model_ft = models.resnet18()#18层的能快点，条件好点的也可以选152
print(model_ft)

#  把模型输出层改成自己的
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    #     model_ft模型的名字
    #     pretrained=use_pretrained，用人家的参数进行初始化
    set_parameter_requires_grad(model_ft, feature_extract)
    #    读取参数，但是都不更新了
    num_ftrs = model_ft.fc.in_features
    #     num_ftrs 全连接层的上一层数据，这里就是512
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 类别数自己根据自己任务来
    #     num_classes=102
    # 重新定义fc层 ，覆盖原有的fc层       ，自己定义的fc是使用反向传播的
    input_size = 64  # 输入大小根据自己配置来

    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

#GPU还是CPU计算
model_ft = model_ft.to(device)

# 模型保存，名字自己起
filename='best.pt'
# 保存网络结构图，和模型里边所有的权重参数保存到本地

# 是否训练所有层
params_to_update = model_ft.parameters()

print("Params to learn:")
# 如果 feature_extract 为 True，则只打印需要更新的参数；否则，打印所有需要更新的参数。
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
#             需要的话再往里边传数据
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

print(model_ft)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)#要训练啥参数，你来定
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
# 学习率衰减，随着epoch的进行，结果会越来越好，降低学习率，使结果更精确
# 以固定的间隔（每10个epoch）将学习率缩小为当前值的10%。这是为了在训练过程中逐渐减小学习率，以帮助模型在训练后期更好地收敛。
criterion = nn.CrossEntropyLoss()
# 交叉熵损失函数

def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, filename='best.pt'):
    since = time.time()
    best_acc = 0
    model.to(device)
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    # 新增：创建一个列表来存储每个epoch的数据
    epoch_data = []

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
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 新增：记录每个epoch的数据
            epoch_data.append({
                'epoch': epoch,
                'phase': phase,
                'loss': epoch_loss,
                'accuracy': epoch_acc.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()

    # 新增：将记录的数据保存为CSV文件
    df = pd.DataFrame(epoch_data)
    df.to_csv('training_results.csv', index=False)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    val_acc_history = [acc.cpu().item() for acc in val_acc_history]
    train_acc_history = [acc.cpu().item() for acc in train_acc_history]
    valid_losses = [loss for loss in valid_losses]  # 这些可能已经是标量
    train_losses = [loss for loss in train_losses]  # 这些可能已经是标量
    LRs = [lr for lr in LRs]  # 这些应该已经是标量
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

import pandas as pd
import matplotlib.pyplot as plt

def save_results_to_csv(train_acc_history, val_acc_history, train_losses, valid_losses, LRs, filename):
    results_df = pd.DataFrame({
        'epoch': range(len(train_acc_history)),
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'train_loss': train_losses,
        'val_loss': valid_losses,
        'learning_rate': LRs[:-1]  # LRs可能比其他列多一个值
    })
    results_df.to_csv(filename, index=False)

def plot_training_results(train_acc_history, val_acc_history, train_losses, valid_losses, LRs, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(train_acc_history, label='Training Accuracy')
    ax1.plot(val_acc_history, label='Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.set_title('Model Accuracy')

    ax2.plot(train_losses, label='Training Loss')
    ax2.plot(valid_losses, label='Validation Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.set_title('Model Loss')

    ax3.plot(LRs[:-1])
    ax3.set_ylabel('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_title('Learning Rate')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 第一次训练
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=50)

# 保存第一次训练结果到CSV
save_results_to_csv(train_acc_history, val_acc_history, train_losses, valid_losses, LRs, 'training_results_1.csv')

# 绘制并保存第一次训练的图表
plot_training_results(train_acc_history, val_acc_history, train_losses, valid_losses, LRs, 'training_results_plots_1.png')

# 第二次训练
for param in model_ft.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

criterion = nn.CrossEntropyLoss()

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

model_ft, val_acc_history_2, train_acc_history_2, valid_losses_2, train_losses_2, LRs_2 = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=50)

# 保存第二次训练结果到CSV
save_results_to_csv(train_acc_history_2, val_acc_history_2, train_losses_2, valid_losses_2, LRs_2, 'training_results_2.csv')

# 绘制并保存第二次训练的图表
plot_training_results(train_acc_history_2, val_acc_history_2, train_losses_2, valid_losses_2, LRs_2, 'training_results_plots_2.png')

# 最后的模型评估和可视化
model_ft.eval()

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(device)

# 保存文件的名字
filename='best.pt'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
# images, labels = dataiter.next()
images, labels = next(dataiter)


model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
preds

def im_convert(tensor):
    """ 展示数据"""

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

    pred_class = str(preds[idx] + 1)  # 假设类别从 1 开始
    true_class = str(labels[idx].item() + 1)  # 假设类别从 1 开始

    pred_name = cat_to_name.get(pred_class, "Unknown")
    true_name = cat_to_name.get(true_class, "Unknown")

    ax.set_title("{} ({})".format(pred_name, true_name),
                 color=("green" if pred_name == true_name else "red"))

plt.savefig('./flower_result.png')
plt.show()
