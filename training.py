import torch
from torch import optim
import numpy as np
import tqdm
from tqdm.notebook import tqdm
import losses
import optimizers
from Config import Config

# define function to train NN in each epoch
def train_and_validation_per_epoch(model, criterion, optimizer,scheduler,train_dataloader,valid_dataloader,device):
  
  # trainging phase
  train_losses = []
  train_acc = []

  preds = []
  labels = []

  model.train()
  tbar = tqdm(train_dataloader) # for visualizing

  for step, (batch_img, batch_label) in enumerate(tbar):
    batch_img = batch_img.to(device)
    batch_label = batch_label.to(device)

    optimizer.zero_grad() # clear the gradient
    y_pred = model(batch_img)
    preds.extend(y_pred)
    labels.extend(batch_label)

    # calculating loss
    loss = criterion(y_pred,batch_label) # calculate current loss
    train_losses.append(loss.item())  # save loss
    loss.backward() # perform back propagation
    optimizer.step() # update learning rate
    tbar.set_postfix(loss = loss.item()) # update progress bar

    # calculating accuracy
    train_acc.append((y_pred.argmax(axis=-1) == batch_label).sum().item() / y_pred.shape[0])

  # Validation phase
  validation_losses = []
  val_acc = []

  pred = []
  labels = []

  model.eval() # switch on validation mode
  tbar = tqdm(valid_dataloader)

  for step, (batch_img, batch_label) in enumerate(tbar):
    batch_img = batch_img.to(device)
    batch_label = batch_label.to(device)

    with torch.no_grad():   # no gradient duting validation phase
      y_pred = model(batch_img)
      
    preds.extend(y_pred)
    labels.extend(batch_label)
    loss = criterion(y_pred,batch_label)
    validation_losses.append(loss.item())
    tbar.set_postfix(loss=loss.item())

    # calculating accuracy
    val_acc.append((y_pred.argmax(axis=-1) == batch_label).sum().item() / y_pred.shape[0])

  return np.mean(train_losses), np.mean(validation_losses), np.mean(train_acc), np.mean(val_acc)


def train_epochs(model,optimizer, scheduler,train_dataloader,valid_dataloader,epochs,model_path):
  best_score = np.inf
  train_losses = []
  validation_losses = []
  train_accuracies = []
  validation_accuracies = []

  for epoch in range(epochs):
    print(f'........ epoch {epoch}')
    
    train_loss, valid_loss, train_acc, val_acc = train_and_validation_per_epoch(model, 
                                                            losses.criterion(),
                                                            optimizer,
                                                            scheduler,
                                                            train_dataloader,
                                                            valid_dataloader,
                                                            device = Config.DEVICE
                                                            )
    train_losses.append(train_loss)
    validation_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    validation_accuracies.append(val_acc)
    scheduler.step(sum(train_losses)/len(train_losses))
    print(f'For current epoch {epoch}, mean  train_loss: {train_loss}, mean train_acc: {train_acc} , mean val_loss: {valid_loss}, mean vali_acc: {val_acc}')
    if valid_loss < best_score:
      best_score = valid_loss
      torch.save(model.state_dict(), model_path)
  return train_losses, validation_losses, train_accuracies, validation_accuracies
