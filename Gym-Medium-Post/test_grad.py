import torch

device='cuda'
print('cuda version : {}'.format(torch.version.cuda))
# device='cpu'
a = torch.randn(3, dtype=torch.float, requires_grad = True, device=device)
b = torch.randn(3, dtype=torch.float).to(device)

# Post facto set gradients

a.requires_grad_()
b.requires_grad_()
print("a is ",a)
print("b is ",b)

loss1 = a.sum()

loss2 = b.sum()

loss1.backward()

loss2.backward()

print("Gradient wrt to a is ",a.grad)
print("Gradient wrt to b is ",b.grad)

print(a.grad)