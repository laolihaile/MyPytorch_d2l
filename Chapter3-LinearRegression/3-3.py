import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
setup_seed(1)


true_w = torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)

def load_array(data_arrays,batch_size,is_train=True):
    """构造pytorch数据迭代器"""
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size=10
data_iter=load_array((features,labels),batch_size)
#print(next(iter(data_iter)))

net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss=nn.MSELoss()

trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()# 梯度清零
        l.backward() # pytorch已做了sum
        trainer.step()

    l=loss(net(features),labels)
    print(f'epoch {epoch+1},loss {l:f}')

w=net[0].weight.data
print('w的估计误差：',true_w-w.reshape(true_w.shape))
b=net[0].bias.data
print('b的估计误差：',true_b-b)

# epoch 1,loss 0.000172
# epoch 2,loss 0.000105
# epoch 3,loss 0.000104
# w的估计误差： tensor([-0.0002,  0.0002])
# b的估计误差： tensor([7.1526e-05])