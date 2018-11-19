import json
from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import seaborn as sns

class Read_Data:
  def __init__(self, file_name):
    self.file_name = file_name
  
  def open_file(self):
    with open(self.file_name, 'r', encoding='utf-8') as f:
      self.data = json.load(f)
  
  def get_avg_value(self, tipe, subkey, fold, epoch):
    # get value per epoch
    values = []
    for i in range(epoch):
      epoch_values = []

      for j in range(fold):
        value = self.data['fold_{}'.format(j+1)]['epoch_{}'.format(i+1)][tipe][subkey]
        epoch_values.append(value)
      
      avg_epoch = np.mean(epoch_values)
      values.append(avg_epoch)
    
    return values

  def get_matrix(self, tipe, subkey, fold, epoch):
    value = np.empty(shape=(10,10))
    for i in range(fold):
      for j in range(epoch):
        value[i,j] = self.data['fold_{}'.format(i+1)]['epoch_{}'.format(j+1)][tipe][subkey]
    
    return value

  def count_mae(self, actual, predicted):
    # Change Label
    new_actual = np.array(actual)
    new_predicted = np.array(predicted)
    
    new_actual[new_actual==1] = -1
    new_actual[new_actual==2] = 1
    new_actual[new_actual==3] = 2
    new_actual[new_actual==4] = -2

    new_predicted[new_predicted==1] = -1
    new_predicted[new_predicted==2] = 1
    new_predicted[new_predicted==3] = 2
    new_predicted[new_predicted==4] = -2

    idx_to_label = {
      -2:'highly_negative',
      -1:'negative',
      0:'neutral',
      1:'positive',
      2:'highly_positive'
    }

    label_to_idx = {
      'highly_negative':-2,
      'negative':-1,
      'neutral':0,
      'positive':1,
      'highly_positive':2
    }

    label_list = ['highly_negative','negative','neutral','positive','highly_positive']

    mae_temp = []

    for label in label_list:
      jml = 0
      avg_mae = 0

      for i in range(len(actual)):
        label_act = idx_to_label[new_actual[i]]

        if label_act == label:
          delta = abs(new_actual[i]-new_predicted[i])
          avg_mae += delta
          jml += 1
      
      mae_label = avg_mae/jml
      mae_temp.append(mae_label)
    
    mae = np.mean(mae_temp)
    return mae
  
  def count_mae_native(self, tipe, label=None):
    label_predicted_to_idx = {
      'highly_Negative': -2,
      'negative':-1,
      'neutral':0,
      'positive':1,
      'highly_positive':2
    }

    label_actual_to_idx = {
      'Highly_Negative': -2,
      'Negative':-1,
      'Neutral':0,
      'Positive':1,
      'Highly_Positive':2
    }

    label_list = ['Highly_Negative','Negative','Neutral','Positive','Highly_Positive']
    predicted_list = ['highly_Negative','negative','neutral','positive','highly_positive']

    mae_fold = []
    for fold in range(1,11):

      mae_temp = []
      for label_actual in label_list:
        jml=0
        avg_mae=0
        for label_predicted in predicted_list:
          value = self.data['fold_{}'.format(fold)]['epoch_10']['predicted']['testing'][label_actual]['detail_predict'][label_predicted]
          delta = abs((label_actual_to_idx[label_actual]-label_predicted_to_idx[label_predicted])*value)
          avg_mae+=delta
          jml+=value
        
        if jml is not 0:
          mae_label = avg_mae/jml
          mae_temp.append(mae_label)
      
      mae_epoch = np.mean(mae_temp)
      mae_fold.append(mae_epoch)
      # print(mae_epoch)
    
    mae = np.mean(mae_fold)
    if tipe is 'fold':
      return mae_fold
    else:
      return mae
  
  def count_mae_epoch(self):
    label_predicted_to_idx = {
      'highly_Negative': -2,
      'negative':-1,
      'neutral':0,
      'positive':1,
      'highly_positive':2
    }

    label_actual_to_idx = {
      'Highly_Negative': -2,
      'Negative':-1,
      'Neutral':0,
      'Positive':1,
      'Highly_Positive':2
    }

    label_list = ['Highly_Negative','Negative','Neutral','Positive','Highly_Positive']
    predicted_list = ['highly_Negative','negative','neutral','positive','highly_positive']

    macro_mae = []
    for epoch in range(1,11):

      mae_epoch = []
      for fold in range(1,11):

        # count mae
        mae_temp = []
        for label_actual in label_list:
          jml=0
          avg_mae=0
          for label_predicted in predicted_list:
            value = self.data['fold_{}'.format(fold)]['epoch_{}'.format(epoch)]['predicted']['testing'][label_actual]['detail_predict'][label_predicted]
            delta = abs((label_actual_to_idx[label_actual]-label_predicted_to_idx[label_predicted])*value)
            avg_mae+=delta
            jml+=value
          
          if jml is not 0:
            mae_label = avg_mae/jml
            mae_temp.append(mae_label)
        
        mae_fold = np.mean(mae_temp)
        mae_epoch.append(mae_fold)
        # print(mae_epoch)
    
      macro_mae.append(np.mean(mae_epoch))

    return macro_mae
  

# file = Read_Data('output training/json/v5/sentimen_own_full_repaired_non-static_own-model2_subtaskC_[3, 4, 5]_[100, 100, 100].json')
# file.open_file()
# mae = file.count_mae_native('average')
# print('mae', mae)
    
    

    
