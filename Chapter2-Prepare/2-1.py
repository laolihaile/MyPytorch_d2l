# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch

x=torch.arange(12)
print(x[2:4]) #zuo bi you kai
print(x.shape)
print(x.numel())
X=x.reshape(3,4)
X=x.reshape(3,-1)
X=x.reshape(-1,4)
print(X)
Y=torch.ones((2,3,4))
Z=torch.zeros((2,3,4))
X=torch.randn((3,4))
torch.tensor([[1,2,3],[2,3,4],[3,4,5]])

x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])

x+y,x-y,x*y,x/y,x**y
torch.exp(x)

X = torch.arange(12,dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0,1,4,5],[1,2,3,4],[4,3,2,1]])
Z=torch.cat((X,Y),dim=0) # 按行合并，行数增加，列数不变
Z = torch.cat((X,Y),dim=1) # 按列合并，列数增加，行数不变
Z =torch.cat((X,Y),dim=-1)
print(Z)
X==Y

a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
a+b

# ijsheng

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
