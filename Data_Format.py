# -*- coding: utf-8 -*-
import torch as torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Data_Format_1:
    
    def __init__(self, DATA):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        x_axis = DATA[["sex"]].copy()
    
        #convert sex to int type
        arr = []
        for i in x_axis["sex"]:
          if i == "Female":
            arr.append(0)
          else:
            arr.append(1)
        #append to all data
        x_axis["sex"] = arr
        #add all remaining integer values
        x_axis["age"] = DATA[["age"]].copy()
        #set y axis as the two year recidivism
        y_axis = DATA[["two_year_recid"]].copy()

        #move data to the device fqewf
        self.new_x_axis = torch.tensor(x_axis.values)
        self.new_y_axis = torch.tensor(y_axis.values)
        #find the index at which to split the data
        val = np.floor(len(self.new_x_axis)*0.8)
    
        #first 80% of the data
        self.x_train = self.new_x_axis[:val.astype(int)].to(device)
        self.y_train = self.new_y_axis[:val.astype(int)].to(device)
        
        #last 20% of the data
        self.x_test = self.new_x_axis[val.astype(int):].to(device)
        self.y_test = self.new_y_axis[val.astype(int):].to(device)

class Data_Format_2:
    def __init__(self, DATA):
        arr = []
        charge_description = DATA[["c_charge_desc"]].copy()
        
        for i in charge_description["c_charge_desc"]:
          arr.append(i)
        
        arr = np.array(arr)
        
        #Label Encoder will turn these values into integer values
        int_vals = LabelEncoder().fit_transform(arr)
        
        #One Hot Encoder will create the respective one hot vectors
        one_hot_vector = OneHotEncoder(sparse = False).fit_transform(int_vals.reshape(7214, 1))
        x2_axis = DATA[["sex"]].copy()

        #convert sex to int type
        arr = []
        for i in x2_axis["sex"]:
          if i == "Female":
            arr.append(0)
          else:
            arr.append(1)

        x2_axis = DATA[["c_charge_degree"]].copy()
        
        #convert degree to int type
        arr2 = []
        for i in x2_axis["c_charge_degree"]:
          if i == "F":
            arr2.append(0)
          else:
            arr2.append(1)
        #append to all data
        x2_axis["sex"] = arr
        x2_axis["c_charge_degree"] = arr2
        #get all other integer valued data
        x2_axis[["age", "juv_fel_count","juv_misd_count", "priors_count"]] = DATA[["age", "juv_fel_count","juv_misd_count", "priors_count"]].copy()
        y2_axis = DATA[["two_year_recid"]].copy()
        
        #get race from data
        
        race = DATA[["race"]].copy()
        
        #convert race to int type
        arr3 = []
        for i in race["race"]:
          if i == "African-American":
            arr3.append(0)
          elif i == "Caucasian":
            arr3.append(1)
          else:
            arr3.append(-1)
        
        #append to all data
        race["race"] = arr3
        
        x2_axis = x2_axis.to_numpy()
        
        empty_arr = np.empty((7214, 444))

        for i in range(len(x2_axis)):
          empty_arr[i] = np.concatenate((x2_axis[i], one_hot_vector[i]))
          
        #split data based on paper: 80% training and 20% test
        self.new_x_axis = torch.from_numpy(empty_arr)
        self.new_y_axis = torch.tensor(y2_axis.values)
        
        #find the indices at which to split the data
        val = np.floor(len(self.new_x_axis)*0.8)
        
        #first 80% of the data
        self.x_train = self.new_x_axis[:val.astype(int)]
        self.y_train = self.new_y_axis[:val.astype(int)]
        
        #last 20% of the data
        self.x_test = self.new_x_axis[val.astype(int):]
        self.y_test = self.new_y_axis[val.astype(int):]
        
        race_data = torch.tensor(race.values)

        #find the indices at which to split the data
        val = np.floor(len(race_data)*0.8)
        
        self.racetrain_dataset = race_data[:val.astype(int)]
        self.racetest_dataset = race_data[val.astype(int):]