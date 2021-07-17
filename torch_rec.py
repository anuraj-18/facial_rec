from torch import nn, tensor, randn, float as flt, device, cuda
from torch.nn import functional as F
from torch import optim
import cv2, numpy as np 
import time

device_name="cuda" if cuda.is_available() else "cpu"
dtype=flt
device=device(device_name)
#print(device)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.denseN=nn.Sequential(
            nn.Linear(480*640*3, 30),
            nn.Sigmoid(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.Sigmoid()
            )
    def forward(self, x):
        x=self.denseN(x)
        return x

neural_net=NeuralNet()
neural_net.to(device=device)
optimizer=optim.Adam(neural_net.parameters(), lr=1e-3)

"""
Types of optims:
    SGD
    Adagrad
    RMSprop
    Adam
"""
x=[]
target=[]
img=""
for i in range(12,22):
    name = "images/shubham"+str(i)+".jpg"
    arr = cv2.imread(str(name)) # 640x480x3 array
    arr = np.reshape(arr,(1,480*640*3))/255
    x.append(arr[0])
    img=arr
    target.append([1,0,0,0])

x=tensor(x, dtype=dtype, device=device)
target = tensor(target, dtype=dtype, device=device)
print(x.shape, target.shape)
loss_fn=nn.MSELoss()

"""
Types of losses:
CrossEntropyLoss
MSELoss
L1Loss
"""
start=time.time()
for i in range(401):
    res=neural_net.forward(x)

    optimizer.zero_grad()

    loss=loss_fn(res, target)
    loss.backward()

    if i%100==0:
        print(loss.item())

    optimizer.step()
end=time.time()
print("Time taken",end-start)
img=tensor(img, dtype=dtype, device=device)
print(neural_net.forward(img).argmax())
