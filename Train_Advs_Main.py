# -*- coding: utf-8 -*-
import torch

# training loop
def Train_Adv(epochs, network_model, x_train,racetrain_dataset, adversary_model, adversary_optimizer, adversary_criterion):
    for epoch in range(epochs):
            adversary_optimizer.zero_grad()           # cleans the gradients    
    
            a = torch.sigmoid(network_model(x_train.float()))            # forward pass
            b = adversary_model(a)            # forward pass
      
            adversary_loss = adversary_criterion(b, racetrain_dataset.float())  # compute the loss and perform the backward pass
            adversary_loss.backward()                 # computes the gradients
      
            adversary_optimizer.step()                # update the parameters
    
            if epoch % 20 == 0:
              print("epoch:", epoch, "loss=", adversary_loss.item())
              
def Train_Main(epochs, network_model,x_train,y_train, network_optimizer, network_criterion, adversary_model):

    # training loop
    for epoch in range(epochs):
      network_optimizer.zero_grad()
    
      a = torch.sigmoid(network_model(x_train.float())) 
      b = adversary_model(a)
    
      network_loss = network_criterion(a, y_train.float())
      network_loss.backward(retain_graph = True)
      
      network_optimizer.step()
    
      if (epoch) % 20 == 0: 
        print("epoch:", epoch, "\tNetwork loss=", network_loss.item())
        
    return a