import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display() #用svg显示图片清晰度高一些
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=False)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=False)
print(len(mnist_train),len(mnist_test))
print(mnist_train[0][0].shape)
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def get_dataloader_workers():#建议此机器最大进程为20
    """使用4个进程来读取数据。"""
    return 4

batch_size=256
train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

timer=d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop()}:.2f sec')
