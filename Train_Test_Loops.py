# -*- coding: utf-8 -*-

import torch

def Training_Loop(model, criterion, optimizer, epochs, trainloader):
    epochs = 1000
    # training loop
    for epoch in range(epochs):
        for i, (x_i, y_i) in enumerate(trainloader):
            optimizer.zero_grad()           # cleans the gradients   
            y_hat_i = model(x_i.float())            # forward pass
            loss = criterion(y_hat_i, y_i.float())  # compute the loss and perform the backward pass
            loss.backward()                 # computes the gradients
            optimizer.step()                # update the parameters
    
        if epoch % 20 == 0:
          print("epoch:", epoch, "loss=", loss.item())
          
def Testing_Loop(model, criterion, testloader):
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        for k, (x_k, y_k) in enumerate(testloader):
            y_hat_k = model(x_k.float())
            loss_test = criterion(y_hat_k, y_k.float())
            total_loss += float(loss_test)
    
    print(total_loss)
