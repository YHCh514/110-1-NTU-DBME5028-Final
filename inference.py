# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:32:37 2022

@author: sammy
"""

#testing
#load data
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
from myGetData import GetTestDataSet
from myModel import SimSiam

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
args = parser.parse_args()
print(args.datapath)

##final testing for submission
queries = pd.read_csv(os.path.join(args.datapath, 'queries.csv'),header=None,)
kaggle_paths=[]
for i in range (len(queries[0])):
    kaggle_paths.append([queries[0][i][-16:-4],queries[1][i][-16:-4]])
kaggle_dataset = GetTestDataSet(kaggle_paths,args.datapath,kaggle = True)
kaggle_val_loader = DataLoader(dataset=kaggle_dataset, batch_size=1, shuffle=False)

#model
model = SimSiam()
# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Move model to the device specified above
model.to(device)

#upload the checkpoint file
state_dict = torch.load('checkpoint.pth')

#load the state dictionary to model
model.load_state_dict(state_dict)

#testing
#evaluation mode
model.eval()
result = []  #storing the result of prediction (probability of being positive/abnormal)
count = 0
with torch.no_grad(): #don't calculate the gradient
    for input1, input2 in kaggle_val_loader:
        # Move to device
        input1, input2 = input1.to(device), input2.to(device)
        # Forward pass
        loss = model.forward_test(input1, input2)
        count += 1
        print(count,'/',len(kaggle_val_loader))
        #store the result
        result.append(loss.to('cpu'))

result = np.array(result)

#build data frame for pandas library
submission = []
for i in range (len(queries)):
    submission.append(queries[0][i][0:12]+'_'+queries[1][i][0:12])
#for i in range (len(submission['query'])):
    #submission['query'][i] = unicode(submission['query'][i], "utf-8")

output_r = []
th = -0.4
for i in range (len(result)):
    if result[i] < th:
        output_r.append(int(1))
    else: output_r.append(int(0))

df = pd.DataFrame({'query':submission,
                      'prediction':output_r})
#construct csv file for submission
df.to_csv(os.path.join(args.datapath, 'prediction.csv'),index=False)