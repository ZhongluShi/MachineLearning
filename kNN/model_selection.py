# train_test_split
import numpy as np 

def train_test_split(X,y,test_ratio,seed=None):
  '''将数据集X和y，按照test_ratio拆分成训练集X_train、y_train和测试集X_test、y_test'''

  assert X.shape[0] == y.shape[0], 'the number of X must equal to the number of y'
  assert 0.0 <= test_ratio <= 1.0,'test_ratio must be at least 0 and at most 1'
  
  if seed:
    np.random.seed(seed)

  shuffle_indexes = np.random.permutation(len(X))
  test_size = int(test_ratio * len(X))
  test_index = shuffle_indexes[:test_size]
  train_index = shuffle_indexes[test_size:]
  X_test = X[test_index]
  y_test = y[test_index]
  X_train = X[train_index]
  y_train = y[train_index]
  return X_train,X_test,y_train,y_test

  