import os
import torch
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
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=5)

for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 加载之前训练好的权重参数

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=5,)

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
fig=plt.figure(figsize=(20, 20))
columns =4
rows = 2

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])]==cat_to_name[str(labels[idx].item())] else "red"))

plt.savefig('./flower_result.png')
plt.show()
