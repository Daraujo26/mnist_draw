from tqdm import tqdm
import torch
import numpy as np
import gradio as gr
import torchvision.datasets
from torchvision.transforms import ToTensor

ds = torchvision.datasets.MNIST(".", train=True, download=True, transform=ToTensor())     
x_train = torch.zeros(len(ds), 28, 28, dtype=torch.float32)
y_train = torch.zeros(len(ds), dtype=torch.int64)
for i, (x, y) in tqdm(enumerate(ds)):
    x_train[i] = x
    y_train[i] = y

x_train = x_train.to("cpu").flatten(start_dim=1, end_dim=2)

def func(l):
    k = len(l)
    temp = {}
    for i in range(k):
        if l[i] in temp.keys():
            temp[l[i]] += 1
        else:
            temp[l[i]] = 1
    return list(temp.keys())[np.argmax(list(temp.values()))]

def predict(x, k=3):
    if x is None: return "[Empty]"
    x = torch.Tensor(x / 255).to("cpu").flatten()

    dist = torch.sqrt(torch.sum((x_train - x) ** 2, dim=1))
    
    value = []
    index = []
    for i in range(k):
        val, ind = torch.min(dist, dim=0)
        value.append(val.item())
        index.append(ind.item())
        dist[ind] = 10000

    labels = func(y_train[torch.tensor(index)].tolist())

    return labels

gr.Interface(fn=predict, 
             inputs="sketchpad",
             outputs="label",
             live=True).launch()
