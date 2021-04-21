## 平行化代码

```python
    #别的数据模型都放到cuda:0作为主gpu就行了
    resnet34 = tv.models.resnet34(pretrained=True)
    resnet34.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
    # print(resnet34)
    resnet34.fc = nn.Linear(512, 10)
    # # print(resnet34)
    resnet34 = nn.DataParallel(resnet34, device_ids=[0, 1, 2, 3])
    resnet34.to(device)
```

## 问题

### 描述

服务器多gpu转单机单cpu出现问题

```python
    parallel = t.load('resnet34FationMnist.pkl', map_location=lambda storage, loc: storage.cuda(0)) # parallel obj

    resnet34 = parallel.module # -> nn.module
    resnet34.to(device)
```

