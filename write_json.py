import json
from collections import defaultdict

class Write:
  
  def __init__(self, file_name, output):
    self.file_name = file_name
    self.output = dict(output)
  
  def set_value(self, key, value):
    self.output[key] = value
  
  def write_to_file(self):
    with open(self.file_name, 'w') as outfile:
      json.dump(self.output, outfile, indent=4)