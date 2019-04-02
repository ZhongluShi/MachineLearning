import numpy as np

def accuracy_score(y_test,y_predict):
  '''计算y_predict相对于y_test的准确度'''
  assert y_test.shape[0] == y_predict.shape[0],'the size of y_test must equal to the size of y_predict'
  return sum(y_predict == y_test) / len(y_test)

