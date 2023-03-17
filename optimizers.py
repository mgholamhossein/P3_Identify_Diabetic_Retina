import torch
from torch.optim import optimizer

def def_optimize(model):
  optimizer = torch.optim.Adam(model.parameters(),lr=5e-6) #lr=5e-6
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=3,verbose=True)
  return (optimizer, scheduler)