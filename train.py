# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:24:59 2022

@author: sammy
"""

#load data
import os
import glob
import torch
from torch import optim
from torch.utils.data import DataLoader
import argparse
from myGetData import GetDataSet
from myModel import SimSiam

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
args = parser.parse_args()

##input data
train_paths = glob.glob(os.path.join(os.path.join(args.datapath, 'train'), '*.png'))
train_dataset = GetDataSet('train',train_paths)

#construct data loader
import torch.utils.data as data
batch_size = 16
ratio = 0.2
size_train = int(len(train_dataset) * (1 - ratio))
size_val = len(train_dataset)-size_train
train_set, valid_set = data.random_split(train_dataset, [size_train,size_val])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=True)

model = SimSiam()

# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Move model to the device specified above
model.to(device)

#set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.parameters(),lr = 0.0001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#training
epochs = 20
best_valid_loss = 10
best_epoch = 0

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for input1, input2, input3 in train_loader:
        # Move to device
        input1, input2, input3 = input1.to(device), input2.to(device), input3.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        loss = model.forward(input1, input2, input3)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*input1.size(0)
        
        # Print the progress of our training
        #counter += 1
        #print(counter, "/", len(train_loader))
    
    scheduler.step()
    
     # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for input1, input2, input3 in val_loader:
            # Move to device
            input1, input2, input3 = input1.to(device), input2.to(device), input3.to(device)
            # Forward pass
            loss = model.forward(input1, input2, input3)
            # Add loss to the validation set's running loss
            val_loss += loss.item()*input1.size(0)
            
            # Print the progress of our evaluation
            #counter += 1
            #print(counter, "/", len(val_loader))  
            
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    torch.save(model.state_dict(), 'checkpoint_'+str(epoch+1)+'.pth')
    if (valid_loss<best_valid_loss):
        best_valid_loss = valid_loss
        best_epoch = epoch
    # Print out the information
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))
    print('Best Epoch: {} \t Best Validation Loss: {:.6f}'.format(best_epoch, best_valid_loss))