import torch
x=torch.arange(4.0,requires_grad=True)
x.requires_grad_(True)
print(x.grad)
y=2*torch.dot(x,x)
print(y)
y.backward()
print(x.grad)
print(x.grad==4*x)
#x.grad.zero_()
#print(x.grad)
y=x.sum()
y.backward()
y=2*torch.dot(x,x)
y.backward()
print(x.grad)

x.grad.zero_()
y=x*x
y.sum().backward()
print(x.grad)

x.grad.zero
y=x*x
u=y.detach()
print(y)
print(u)
z=u*x
z.sum().backward()
print(x.grad)