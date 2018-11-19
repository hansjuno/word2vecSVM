import model
import prepare_data
import read_embeddings
import write_json

import argparse
import inspect
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data

from collections import defaultdict
from math import ceil
import math

import matplotlib
import matplotlib as plt

import time
import pprint


parser = argparse.ArgumentParser(
    description='Sentiment Analysis CNN With Word Embeddings')
parser.add_argument('-cuda', type=bool, default=True, help='whether to use GPU')
parser.add_argument('-batch_size', type=int, default=50, help='size of minibatches')
parser.add_argument('-embeddings_source', type=str, default='default', help='only default')
parser.add_argument('-embeddings_mode', type=str, default='non-static', help='either random and pre trained')
parser.add_argument('-embeddings_dim', type=int, default=300, help='default size of embeddings')
parser.add_argument('-epoch_num', type=int, default=10, help='number of epochs')
parser.add_argument('-conv_mode', type=str, default='wide', help='convolution mode')
parser.add_argument('-fold_num', type=int, default=10, help='number of folding')
parser.add_argument('-subtask', type=str, default='substakC', help='substakA, B, C')
parser.add_argument('-kernel_width', type=int, default=10, help='region size')
parser.add_argument('-feature_num', type=int, default=100, help='100, 200, 300, 400')


args = parser.parse_args()

def cross_validate(fold, data, embeddings, args):
  actual_counts    = defaultdict(int)
  predicted_counts = defaultdict(int)
  match_counts     = defaultdict(int)

  split_width = int(ceil(len(data.examples)/fold))

  for i in range(fold):
    print('FOLD [{}]'.format(i + 1))

    train_examples = data.examples[:] 
    del train_examples[i*split_width:min(len(data.examples), (i+1)*split_width)]
    test_examples = data.examples[i*split_width:min(len(data.examples), (i+1)*split_width)]

    total_len = len(data.examples)
    train_len = len(train_examples)
    test_len  = len(test_examples)

    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)

    for example in train_examples:
      train_counts[example.label] += 1

    for example in test_examples:
      test_counts[example.label] += 1

    output['fold_{}'.format(i+1)]['total_examples'] = total_len
    output['fold_{}'.format(i+1)]['detail_examples'] = []
   
    # Training Label
    high_pos = train_counts['Highly_Positive']
    pos = train_counts['Positive']
    neg = train_counts['Negative']
    high_neg = train_counts['Highly_Negative']
    neu = train_counts['Neutral']
    output['fold_{}'.format(i+1)]['detail_examples'].append({
      'examples': 'train',
      'total': train_len,
      'high_positive': high_pos,
      'positive': pos,
      'neutral': neu,
      'negative': neg,
      'high_negative': high_neg,
    })

    # Testing Label
    high_pos = test_counts['Highly_Positive']
    pos = test_counts['Positive']
    neg = test_counts['Negative']
    high_neg = test_counts['Highly_Negative']
    neu = test_counts['Neutral']

    output['fold_{}'.format(i+1)]['detail_examples'].append({
      'examples': 'test',
      'total': test_len,
      'high_positive': high_pos,
      'positive': pos,
      'neutral': neu,
      'negative': neg,
      'high_negative': high_neg,
    })
    
    fields = data.fields
    train_set = torchtext.data.Dataset(examples=train_examples, fields=fields)
    test_set  = torchtext.data.Dataset(examples=test_examples, fields=fields)

    # Building the vocabs
    text_field  = None
    label_field = None

    for field_name, field_object in fields:
      if field_name == 'text':
        text_field = field_object
      elif field_name == 'label':
        label_field = field_object
    
    text_field.build_vocab(train_set)
    label_field.build_vocab(train_set)

    data.vocab_to_idx = dict(text_field.vocab.stoi)
    data.idx_to_vocab = {v: k for k, v in data.vocab_to_idx.items()}

    data.label_to_idx = dict(label_field.vocab.stoi)
    data.idx_to_label = {v: k for k, v in data.label_to_idx.items()}

    embed_num = len(text_field.vocab)
    label_num = len(label_field.vocab)

    # Loading pre-trained embeddings
    emb_init_values = np.array(data.create_folds_embeddings(embeddings, args))

    train_iter, test_iter = torchtext.data.Iterator.splits((train_set, test_set),batch_sizes=(args.batch_size, len(test_set)), device=-1, repeat=False)
    
    # Ini gunanya untuk mengukur training performance dari model
    # (lawannya generalization performance, forget what it's called)
    train_bulk_dataset = train_set,
    train_bulk__size   = len(train_set),
    train_bulk_iter    = torchtext.data.Iterator.splits(datasets=train_bulk_dataset, 
                                                        batch_sizes=train_bulk__size,
                                                        device=-1, 
                                                        repeat=False)[0]
    
    kim2014 = model.CNN(embed_num, label_num -1, args.embeddings_dim, args.embeddings_mode, emb_init_values, args)

    if args.cuda:
      kim2014.cuda()
    
    trained_model = train(kim2014, train_iter, test_iter, data.label_to_idx, data.idx_to_label, train_bulk_iter, args, i)
  
  output['embed_num'] = kim2014.embed_num
  output['label_num'] = kim2014.label_num
  output['embed_dim'] = kim2014.embed_dim
  output['embed_mode'] = kim2014.embed_mode
  output['channel_in'] = kim2014.channel_in
  output['feature_num'] = kim2014.feature_num
  output['kernel_width'] = kim2014.kernel_width
  output['dropout_rate'] = kim2014.dropout_rate
  output['norm_limit'] = kim2014.norm_limit

def train(model, train_iter, test_iter, label_to_idx, idx_to_label, train_bulk_iter, args, fold):
 
  parameters = filter(lambda p: p.requires_grad, model.parameters())

  # Optimizer
  optimizer = torch.optim.Adadelta(parameters) # Adam, Adadelta
  
  # using GPU
  if args.cuda:
      model.cuda()
      
  model.train()

  for epoch in range(1, args.epoch_num+1):
    print("FOLD/EPOCH [{}/{}]".format(fold+1, epoch))
    step = 0
    corrects_sum = 0

    for batch in train_iter:
      text_numerical, target = batch.text, batch.label
      if args.cuda:
        text_numerical, target = text_numerical.cuda(), target.cuda()
      
      text_numerical.data.t_()
      target.data.sub_(1)

      optimizer.zero_grad()

      forward = model(text_numerical)
      
      loss = F.cross_entropy(forward, target)
      loss.backward()

      optimizer.step()
      step += 1

      corrects = (torch.max(forward, 1)[1].view(target.size()).data == target.data).sum()

      accuracy = 100.0 * corrects / batch.batch_size

    # print('--- PEFORMANCE ON TRAIN DATA ---')

    actual, predicted = evaluate(model, train_bulk_iter, 'training', epoch, fold)
    calculate_predict_value(actual, predicted, label_to_idx, idx_to_label, args, 'training', fold, epoch)
    epoch_actual_counts, epoch_predicted_counts, epoch_match_counts, epoch_mae_calculate, epoch_std_calculate = calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label)
    calculate_and_display_SemEval_metrics(epoch_actual_counts, epoch_predicted_counts, epoch_match_counts, epoch_mae_calculate, epoch_std_calculate, args, 'training', epoch, fold)

    # print('--- PEFORMANCE ON TEST DATA ---')

    actual, predicted = evaluate(model, test_iter, 'testing', epoch, fold)
    calculate_predict_value(actual, predicted, label_to_idx, idx_to_label, args, 'testing', fold, epoch)
    epoch_actual_counts, epoch_predicted_counts, epoch_match_counts, epoch_mae_calculate, epoch_std_calculate = calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label)
    calculate_and_display_SemEval_metrics(epoch_actual_counts, epoch_predicted_counts, epoch_match_counts, epoch_mae_calculate, epoch_std_calculate, args, 'testing', epoch, fold)

  return model

def evaluate(model, data_iter, tipe, epoch, fold):

  model.eval()
  corrects, avg_loss = 0, 0

  data_iter.sort_key = lambda x: len(x.text)

  for batch in data_iter:
    
    text_numerical, target = batch.text, batch.label

    if args.cuda:
      text_numerical, target = text_numerical.cuda(), target.cuda()

    text_numerical.data.t_()
    target.data.sub_(1)

    forward = model(text_numerical)
    loss = F.cross_entropy(forward, target, size_average=False)

    avg_loss += loss.data[0]
    corrects += (torch.max(forward, 1)[1].view(target.size()).data == target.data).sum()
  
  size = len(data_iter.dataset)
  avg_loss = avg_loss/size
  accuracy = 100.0 * corrects/size

  # print('Avg Loss = {}'.format(avg_loss))
  output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['avg_loss'] = avg_loss.item()

  if tipe == 'testing':
    if args.cuda:
      data.append(accuracy.item())
    else:
      data.append(accuracy)
  
  return target.data, torch.max(forward, 1)[1].view(target.size()).data

# Menghitung confussion matrix
def calculate_predict_value(actual, predicted, label_to_idx, idx_to_label, args, tipe, fold, epoch):

  assert len(actual)  ==  len(predicted)

  actual_counts   = defaultdict(int)
  predict_counts  = defaultdict(int)

  for i in range(len(actual)):
    idx = actual[i] + 1
    label = idx_to_label[idx.item()]
    predict_counts[label] = defaultdict(int)

  for i in range(len(actual)):
    idx_actual = actual[i] + 1
    label_actual = idx_to_label[idx_actual.item()]
    idx_predict = predicted[i] + 1
    label_predict = idx_to_label[idx_predict.item()]

    label_count = idx_to_label[idx_actual.item()]
    actual_counts[label_count] += 1

    predict_counts[label_actual][label_predict] += 1

  for label in predict_counts.keys():

    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['test_size'] = actual_counts[label]
    
    if args.subtask == 'subtaskA':
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['positive'] = predict_counts[label]['Positive']
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['negative'] = predict_counts[label]['Negative']
      
    elif args.subtask == 'subtaskB':
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['positive'] = predict_counts[label]['Positive']
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['neutral'] = predict_counts[label]['Neutral']
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['negative'] = predict_counts[label]['Negative']

    elif args.subtask == 'subtaskC':
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['highly_positive'] = predict_counts[label]['HighlyPositive']
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['positive'] = predict_counts[label]['Positive']
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['neutral'] = predict_counts[label]['Neutral']
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['negative'] = predict_counts[label]['Negative']
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)]['predicted'][tipe][label]['detail_predict']['highly_Negative'] = predict_counts[label]['HighlyNegative']
      

# Produce counts with the label as the keys
def calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label):

  assert len(actual)  ==  len(predicted)

  fold_actual_counts = defaultdict(int)
  fold_predicted_counts = defaultdict(int)
  fold_match_counts = defaultdict(int)
  mae_calculate_label = defaultdict(int)
  std_calculate_label = defaultdict(float)
    
  mean = 0
  for i in range(len(actual)):
    idx   = actual[i]+1
    mean += idx.item()
  mean = mean/len(actual)
  
  if args.cuda:
    # This made me aware that the train_iter evaluated is only the last batch
    for i in range(len(actual)):
      # the index is incremented since it is decremented prior to processing
      # to account for <unk>, which label is 0
      # Therefore, to match with label dictionaries, the counts are for actual[i] + 1
      idx = actual[i]+1
      idx_predicted = predicted[i]+1

      diff_label = abs((idx_predicted.item())-idx.item())
      diff_mean = (idx.item()-mean)**2

      label = idx_to_label[idx.item()]
      fold_actual_counts[label] += 1
      mae_calculate_label[label] += diff_label
      std_calculate_label[label] += diff_mean

      if actual[i] == predicted[i]:
        fold_match_counts[label] += 1

    for i in range(len(predicted)):
      idx = predicted[i] + 1
      label = idx_to_label[idx.item()]
      fold_predicted_counts[label] += 1
  
  else:
    for i in range(len(actual)):
      # The index is incremented since it is decremented prior to processing
      # to account for <unk>, which label is 0
      # Therefore, to match with label dictionaries, the counts are for actual[i] + 1
      idx = actual[i] + 1

      diff_label = abs((predicted[i]+1)-idx)
      diff_mean = (idx-mean)**2

      label = idx_to_label[idx]
      fold_actual_counts[label] += 1
      mae_calculate_label[label] += diff_label
      std_calculate_label[label] += diff_mean

      if actual[i] == predicted[i]:
        fold_match_counts[label] += 1

    for i in range(len(predicted)):
      idx = predicted[i] + 1
      label = idx_to_label[idx]
      fold_predicted_counts[label] += 1

  return fold_actual_counts, fold_predicted_counts, fold_match_counts, mae_calculate_label, std_calculate_label

def calculate_and_display_SemEval_metrics(actual_counts, predicted_counts, match_counts, mae_calculate_label, std_calculate_label,args, tipe, epoch, fold):
  # Prepare precision, recall, and f-measure for each class
  precisions  = defaultdict(float)
  recalls     = defaultdict(float)
  f_measures  = defaultdict(float)
  macro_mae   = defaultdict(float)
  std_dev     = defaultdict(float)
  klds        = defaultdict(float)
  standard_mae  = 0

  # Mengukur Besar Size dalam Sampel Data
  test_size = sum(actual_counts.values())

  label_count = 0

  for label in actual_counts.keys():

    macro_mae_avg = mae_calculate_label[label] / actual_counts[label] if actual_counts[label] > 0 else 0
    standard_mae  += mae_calculate_label[label]
    
    std_avg = std_calculate_label[label] / (actual_counts[label]-1) if (actual_counts[label]-1) > 0 else 0
    
    kld_actual    = actual_counts[label]/test_size
    kld_predicted = predicted_counts[label]/test_size
    diff = kld_actual/kld_predicted if kld_predicted > 0 else 0
    kld_calculate = kld_actual * math.log(diff) if diff > 0 else 0

    precision = match_counts[label] / predicted_counts[label] if predicted_counts[label] > 0 else 0
    recall    = match_counts[label] / actual_counts[label] if actual_counts[label] > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Store these values in the metric dictionaries
    precisions[label] = precision
    recalls[label]    = recall
    f_measures[label] = f_measure
    std_dev[label]    = std_calculate_label[label]
    klds[label]       = kld_calculate
    macro_mae[label]  = macro_mae_avg

    label_count += 1

    # Print precision, recall, and f-measure for each class
    # print('##--- Calculate Performance ---#')
    # print("On class {} :".format(label))
    
    if args.subtask == 'subtaskA':
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe][label] = [precision, recall, f_measure, kld_calculate, std_avg]
      
      # print("\tPrecision  = {}".format(precision))
      # print("\tRecall     = {}".format(recall))
      # print("\tF1-Score   = {}".format(f_measure))
      # print("\tKLD        = {}".format(kld_calculate))
      # print("\tStandard Deviation   = {}".format(std_avg))
    elif args.subtask == 'subtaskB':
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe][label] = [precision, recall, f_measure, std_avg]
      
      # print("\tPrecision  = {}".format(precision))
      # print("\tRecall     = {}".format(recall))
      # print("\tF1-Score   = {}".format(f_measure))
      # print("\tStandard Deviation   = {}".format(std_avg))
    elif args.subtask == 'subtaskC':
      output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe][label] = [precision, recall, f_measure, macro_mae_avg, std_avg]

      # print("\tPrecision  = {}".format(precision))
      # print("\tRecall     = {}".format(recall))
      # print("\tF1-Score   = {}".format(f_measure))
      # print("\tMacro-MAE  = {}".format(macro_mae_avg))
      # print("\tStandard Deviation   = {}".format(std_avg))

  # Perhitungan Metric Peformance Setiap Subtask
  if args.subtask == 'subtaskA':
    sum_recall = 0
    for label in recalls:
      if (label == "Positive") or (label == "Negative"):
        sum_recall += recalls[label]
    avg_recall = sum_recall / 2

    sum_f_measure = 0
    for label in f_measures:
      if (label == "Positive") or (label == "Negative"):
        sum_f_measure += f_measures[label]
    f_pos_neg = sum_f_measure / 2

    sum_macro_mae = 0
    for label in macro_mae:
      if (label == "Positive") or (label == "Negative"):
        sum_macro_mae += macro_mae[label]
    avg_macro_mae = sum_macro_mae / 2

    sum_std_dev = 0
    for label in std_dev:
      if (label == "Positive") or (label == "Negative"):
        sum_std_dev += std_dev[label]
    avg_std_dev = sum_std_dev/(test_size-1)

    sum_kld = 0
    for label in klds:
      if (label == "Positive") or (label == "Negative"):
        sum_kld += klds[label]
    
    avg_standard_mae  = standard_mae/test_size

  elif args.subtask == 'subtaskB':
    sum_recall = 0
    for label in recalls:
      if (label == "Positive") or (label == "Neutral") or (label == "Negative"):
        sum_recall += recalls[label]
    avg_recall = sum_recall / 3

    sum_f_measure = 0
    for label in f_measures:
      if (label == "Positive") or (label == "Negative"):
        sum_f_measure += f_measures[label]
    f_pos_neg = sum_f_measure / 2

    sum_macro_mae = 0
    for label in macro_mae:
      if (label == "Positive") or (label == "Neutral") or (label == "Negative"):
        sum_macro_mae += macro_mae[label]
    avg_macro_mae = sum_macro_mae / label_count

    sum_std_dev = 0
    for label in std_dev:
      if (label == "Positive") or (label == "Neutral") or (label == "Negative"):
        sum_std_dev += std_dev[label]
    avg_std_dev = sum_std_dev/(test_size-1)

    sum_kld = 0
    for label in klds:
      if (label == "Positive") or (label == "Neutral") or (label == "Negative"):
        sum_kld += klds[label]
    
    avg_standard_mae  = standard_mae/test_size

  elif args.subtask == 'subtaskC':
    sum_recall = 0
    for label in recalls:
      if (label == "HighlyPositive") or (label == "Positive") or (label == "Neutral") or (label == "Negative") or (label == "HighlyNegative"):
        sum_recall += recalls[label]
    avg_recall = sum_recall / 5

    sum_f_measure = 0
    for label in f_measures:
      if (label == "HighlyPositive") or (label == "Positive") or (label == "Negative") or (label == "HighlyNegative"):
        sum_f_measure += f_measures[label]
    f_pos_neg = sum_f_measure / 4

    sum_macro_mae = 0
    for label in macro_mae:
      if (label == "HighlyPositive") or (label == "Positive") or (label == "Neutral") or (label == "Negative") or (label == "HighlyNegative"):
        sum_macro_mae += macro_mae[label]
    avg_macro_mae = sum_macro_mae / label_count

    sum_std_dev = 0
    for label in std_dev:
      if (label == "HighlyPositive") or (label == "Positive") or (label == "Neutral") or (label == "Negative") or (label == "HighlyNegative"):
        sum_std_dev += std_dev[label]
    avg_std_dev = sum_std_dev/(test_size-1)

    sum_kld = 0
    for label in klds:
      if (label == "HighlyPositive") or (label == "Positive") or (label == "Neutral") or (label == "Negative") or (label == "HighlyNegative"):
        sum_kld += klds[label]
    
    avg_standard_mae  = standard_mae/test_size

  # Set Output Dictionary and Print
  if args.subtask == 'subtaskA':
    # print("\nSubtask A measures:")

    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['test_size'] = test_size
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['avg_recall'] = avg_recall
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['f_pos_neg'] = f_pos_neg
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['standar_deviation'] = avg_std_dev
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['kld'] = sum_kld    
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['accuracy'] = sum(match_counts.values()) / test_size

    # print("Test size: {}".format(test_size))
    # print("AvgRecall = {}".format(avg_recall))
    # print("F1-Score  = {}".format(f_pos_neg))
    # print("Standard Deviation = {}".format(avg_std_dev))
    # print("KLD = {}".format(sum_kld))
    print("Accuracy     = {}".format(sum(match_counts.values()) / test_size))

  elif args.subtask == 'subtaskB':
    # print("\nSubtask B measures:")

    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['test_size'] = test_size
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['avg_recall'] = avg_recall
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['f_pos_neg'] = f_pos_neg
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['standar_deviation'] = avg_std_dev
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['accuracy'] = sum(match_counts.values()) / test_size

    # print("Test size: {}".format(test_size))
    # print("AvgRecall = {}".format(avg_recall))
    # print("F1-Score  = {}".format(f_pos_neg))
    # print("Standard Deviation = {}".format(avg_std_dev))
    print("Accuracy     = {}".format(sum(match_counts.values()) / test_size))

  elif args.subtask == 'subtaskC':
    # print("\nSubtask C measures:")

    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['test_size'] = test_size
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['avg_recall'] = avg_recall
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['f_pos_neg'] = f_pos_neg
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['macro_mae'] = avg_macro_mae
    # if avg_macro_mae < 1:
    #   print('Nilai Avg MAE : {}'.format(avg_macro_mae))
    #   exit()
    # print('Nilai MAE : {}'.format(avg_macro_mae))
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['standard_mae'] = avg_standard_mae
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['standar_deviation'] = avg_std_dev
    output['fold_{}'.format(fold+1)]['epoch_{}'.format(epoch)][tipe]['accuracy'] = sum(match_counts.values()) / test_size
    
    print("Test size: {}".format(test_size))
    # print("AvgRecall = {}".format(avg_recall))
    # print("F1-Score  = {}".format(f_pos_neg))
    # print("Macro-MAE    = {}".format(avg_macro_mae))
    # print("Standard-MAE = {}".format(avg_standard_mae))
    # print("Standard Deviation = {}".format(avg_std_dev))
    print("Accuracy     = {}".format(sum(match_counts.values()) / test_size))

if __name__ == '__main__':

  args.model_type = 'kim2014'
  args.embeddings_source = 'own-model1'
  args.embeddings_mode = 'static'
  args.conv_mode = 'None'
  args.subtask = 'subtaskC'
  args.fold_num = 10
  args.epoch_num   = 10  

  data = []

  list_kernel_width = [[6,6,6],[6,7,8],[5,6,7],[4,5,6]]
  list_feature_num = [[100,100,100],[200,200,200],[300,300,300],[400,400,400],[600,600,600]]
  
  for kernel in list_kernel_width:
    for feature in list_feature_num:
      
      try :
        print('Scenario {} {} {}'.format(args.subtask, kernel, feature))
        args.kernel_width = kernel
        args.feature_num = feature
        args.cuda = True
        # Initiate the dictionary
        nested_dict = lambda : defaultdict(nested_dict)
        output = nested_dict()

        output['model_type'] = args.model_type
        output['embedding_source'] = args.embeddings_source
        output['embedding_mode'] = args.embeddings_mode
        output['conv_mode'] = args.conv_mode
        output['folding'] = args.fold_num
        output['epoch'] = args.epoch_num
        output['batch'] = args.batch_size
        output['gpu'] = args.cuda
        output['kernel_width'] = args.kernel_width
        output['feature_num'] = args.feature_num

        print('Started')
        
        
        # Prepare Data
        output['subtask'] = args.subtask
        
        in_data = prepare_data.DataPreparation() 
        in_data.read_dataset(args.subtask, 'json')
        
        
        # Load Embeddings
        output['embedding_source'] = args.embeddings_source

        embeddings = read_embeddings.Embeddings()
        embeddings.read_model(args.embeddings_source)

        print("--- Execution Main.py ---")
        
        # Start Training
        start = time.time()
        cross_validate(args.fold_num, in_data, embeddings, args)
                
        # Training End 
        end = time.time()
        print(' Training done in {} seconds'.format(end-start))

        # Save Ouput
        output['duration'] = end-start
        file_name  = 'output/own_full_repaired_'+args.embeddings_mode+'_'+args.embeddings_source+'_'+args.subtask+'_'+str(args.kernel_width)+'_'+str(args.feature_num)+'.json'

        log_training = write_json.Write(file_name,output)
        log_training.write_to_file()
        print('Sukses Simpan')
      
      except:
        continue


