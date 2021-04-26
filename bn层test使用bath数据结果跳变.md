## BN层

model.train() -> 所有的模型都设置为training = True (针对batchnorm, dropout, instancenorm)
model.eval() -> 所有的模型都training = False

参数 affine是是否使用w和b来学习

参数 track_running_stats 判断是否跟踪 什么情况下都True

参数 training 是是否使用batch的mean和var来进行计算



训练：

track_running_stats True

training True



测试

track_running_stats True

training False



```python
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
from torch import optim
import time

show = ToPILImage()


if __name__ == '__main__':
    seed = 0
    t.manual_seed(seed)
    device = t.device('cuda:0' if t.cuda.is_available() else "cpu")
    # device = 'cpu'
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)),  # 归一化
    ])

    # # 训练集
    # trainset = tv.datasets.FashionMNIST(
    #     root='.',
    #     train=True,
    #     # download=True,
    #     transform=transform)
    #
    # trainloader = t.utils.data.DataLoader(
    #     trainset,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=8,
    #
    # )

    # 测试集
    testset = tv.datasets.FashionMNIST(
        '.',
        train=False,
        # download=True,
        transform=transform)

    testloader = t.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=2)

    classes = ('t-shirt', 'trouser', 'pullover', 'dress',
               'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

    # resnet34 = t.load('resnet34FationMnist.pkl')
    # resnet34.to(device)


    # resnet34 = t.load('resnet34FationMnist.pkl', map_location=lambda storage, loc: storage.cuda(0))
    # resnet34 = t.load('resnet34FationMnist.pkl')


    parallel = t.load('resnet34FationMnist.pkl', map_location=lambda storage, loc: storage.cuda(0)) # parallel obj

    resnet34 = parallel.module # -> nn.module
    resnet34.to(device)
    resnet34.requires_grad_(False)
    # resnet34 = tv.models.resnet34()
    resnet34.bn1.track_running_stats = True
    resnet34.bn1.training = False
    for name, module in resnet34.layer1.named_modules():
        if 'bn' in name:
            print(name)
            module.track_running_stats = True
            # module.affine = False
            module.training = False
            # module.requires_grad_(False)
            # print(next(module.named_parameters()))
    for name, module in resnet34.layer2.named_modules():
        if 'bn' in name:
            module.track_running_stats = True
            module.training = False
            # module.affine = False
            # module.track_running_stats = False
            # module.requires_grad_(False)
    for name, module in resnet34.layer3.named_modules():
        if 'bn' in name:
            module.track_running_stats = True
            module.training = False
            # module.affine = False
            # module.track_running_stats = False
            # module.requires_grad_(False)
    for name, module in resnet34.layer4.named_modules():
        if 'bn' in name:
            module.track_running_stats = True
            module.training = False
            # module.affine = False
            # module.track_running_stats = False
            # module.requires_grad_(False)

    # print(resnet34)
    # print(type(checkpoint.module))
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint.module.items()}
    # resnet34.load_state_dict(state_dict)

    # for param in resnet34.parameters():
    #     print(param.device)

    # images, labels = next(iter(testloader))
    # # #
    # images = images.to(device)
    # labels = labels.to(device)
    # outputs = resnet34(images)
    # _, predicted = t.max(outputs, dim=1)
    # print(predicted)
    # print(outputs.shape)
    # print(labels)


    total = 0
    correct = 0
    count = 0
    with t.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet34(images)
            _, predicted = t.max(outputs, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            count += 1
            print(count)
    print(correct)
    print(total)
    print('10000张测试集中的准确率为: %d %%' % (100 * t.true_divide(correct, total)))
```

