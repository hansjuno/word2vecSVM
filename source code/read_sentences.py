import emot
import html
import os
import re
import json
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np

class Read_Sentences:
  def __init__(self, file_name):
    self.file_name = file_name
    self.emoticons = {}
    self.emojis = {}

  def divide_line(self, line):
    twt_id = line[0]
    polarity = line[1]
    text = line[4]
    
    return twt_id, polarity, text
  
  def divide_line_text(self, line):
    tokens = line.split()
    twt_id = tokens[0]
    polarity = tokens[2]
    text = ' '.join(tokens[5:]).strip('"')

    return twt_id, polarity, text 
  
  def replace_URL(self, string):
    tokens = ['<url>' if '://' in token else token 
                for token in string.split() ]
    return ' '.join(tokens)

  def replace_mention(self, string):
    tokens = [' <mention> ' if token.startswith('@') else token for token in string.split()]
    return ' '.join(tokens)
  
  def replace_mult_occurences(self, string):
    return re.sub(r'(.)\1{2,}', r'\1\1', string)
  
  def replace_ellipsis(self, string):
    return re.sub(r'\.{2,}', ' ', string)
  
  def replace_token_emoticon(self, token):
    if token.startswith('<3'):
      return ' <heart> '

    if token.startswith(':'):
      if token.startswith(":'"):
        if (token == ":')") | (token == ":'))"):
          return ' <tear_smile> '
        if (token == ":'(") | (token == ":'(("):
          return ' <tear_sad> '
      if token == ':D':
          return ' <laugh> '
      if (token == ':P') | (token == ':p'):
          return ' <tongue_out> '
      if (token == ':O') | (token == ':o'):
          return ' <surprised> '
      # if (token == ':/') | (token == ':\\'):
      #     return ' <annoyed> '
      if (token == ':x') | (token == ':*'):
          return ' <kiss> '
      if token == ':3':
          return ' <cat_face> '

    if token.startswith('='):
      if token == '=)':
        return ' <smile> '
      if token == '=))':
        return ' <smile_smile> '
      if (token == '=/') | (token == '=\\'):
        return ' <annoyed> '

    if token.startswith('('):
      if (token == "(':") | (token == "((':"):
        return ' <tear_smile> '

    if token == 'XD':
      return ' <lol> '

    return token
  
  def replace_emoticons(self, string):
    # Smiles
    string = string.replace(':))', ' <smile_smile> ')
    string = string.replace(':)', ' <smile> ')
    string = string.replace(':-))', ' <smile_smile> ')
    string = string.replace(':-)', ' <smile> ')
    string = string.replace('((:', ' <smile_smile> ')
    string = string.replace('(:', ' <smile> ')
    string = string.replace('((-:', ' <smile_smile> ')
    string = string.replace('(-:', ' <smile> ')
    string = string.replace('=))', ' <smile_smile> ')
    string = string.replace('=)', ' <smile> ')
    string = string.replace('^_^', ' <smile> ')

    # Sads
    string = string.replace(':((', ' <sad_sad> ')
    string = string.replace(':(', ' <sad> ')
    string = string.replace(':-((', ' <sad_sad> ')
    string = string.replace(':-(', ' <sad> ')
    string = string.replace(')):', ' <sad_sad> ')
    string = string.replace('):', ' <sad> ')
    string = string.replace('))-:', ' <sad_sad> ')
    string = string.replace(')-:', ' <sad> ')

    # Winks
    string = string.replace(';))', ' <wink_smile> ')
    string = string.replace(';)', ' <wink_smile> ')

    # Tears
    string = string.replace(":'))", ' <tear_smile> ')
    string = string.replace(":')", ' <tear_smile> ')
    string = string.replace(":'((", ' <tear_sad> ')
    string = string.replace(":'(", ' <tear_sad> ')
    string = string.replace("((':", ' <tear_smile> ')
    string = string.replace("(':", ' <tear_smile> ')

    # Some annoyed
    string = string.replace(':/', ' <annoyed> ')
    string = string.replace(':\\', ' <annoyed> ')

    # Straight face
    string = string.replace(':|', ' <straight_face> ')
    string = string.replace(':-|', ' <straight_face> ')

    string = ' '.join([self.replace_token_emoticon(token)
                        for token in string.split()])
    return string

  def detect_emoticons_emojis(self, string):
    emoticons = emot.emoticons(string)
    emojis = emot.emoji(string)
    if len(emoticons) > 0:
      for emoticon in emoticons:
        value = emoticon['value']
        if value != (')' or ':'):
            self.emoticons.setdefault(value, set()).add(string)
    if len(emojis) > 0:
      for emoji in emojis:
        value = emoji['value']
        if value != (')' or ':'):
          self.emojis.setdefault(value, set()).add(string)

  def detect_html(self, string):
    print(html.unescape(string))

  def clean_str(self, string):
    # Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    # with modification
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

    string = re.sub(r"amp;", " ", string)
    string = re.sub(r"gt;", " ", string)
    string = re.sub(r"xa0", " ", string)
    string = re.sub(r"quot", " ", string)
    string = re.sub(r"\b'\b", "", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"#", " <hash_tagh> ", string)
    string = re.sub(r"\[", " [ ", string)
    string = re.sub(r"\]", " ] ", string)
    string = re.sub(r"\'m ", " \'m ", string)
    string = re.sub(r"\'s ", " \'s ", string)
    string = re.sub(r"\'re ", " \'re ", string)
    string = re.sub(r"\'ll ", " \'ll ", string)
    string = re.sub(r"\'d ", " \'d ", string)
    string = re.sub(r"\'ve ", " \'ve ", string)
    string = re.sub(r"n\'t ", " n\'t ", string)
    string = re.sub(r"[^A-Za-z<>!.,]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string

class ReadingIndonesianPolitic (Read_Sentences):

  def __init__(self, filename, tipe):
    self.file_name = filename
    self.tipe = tipe

  # Clear data with label 0 for subtask A
  def clear_data(self, twt_id, polarity, text):
    if polarity is not 0:
      return twt_id, polarity, text

  # Read json data with pandas 
  def read_json(self):
    with open(self.file_name,'r', encoding='utf-8') as file:
      dataset = json.load(file)
    df = pd.io.json.json_normalize(data=dataset, record_path=['RECORDS'])
    data_matrix = df.as_matrix()
    return data_matrix

  def __iter__(self): 
    
    if self.tipe == 'json':
      data = self.read_json() 
    elif self.tipe == 'txt':
      with open(self.file_name, 'r', encoding='utf8', errors='ignore') as txt_file:
        data = txt_file

    # Start Processing
    for line in data:
      if self.tipe =='json':
        twt_id, polarity, text = self.divide_line(line)
      elif self.tipe =='txt':
        twt_id, polarity, text = self.divide_line_text(line)

      text = self.replace_URL(text)
      text = html.unescape(text)
      text = text.replace('""', ' <kutip> ') # text.replace('""', ' <double_quotes> ')
      text = self.replace_mention(text)
      text = self.replace_mult_occurences(text)
      text = text.replace('..', ' <elipsis> ')
      text = self.replace_emoticons(text)
      text = self.clean_str(text)
      text = text.lower()

      yield twt_id, polarity, text  

