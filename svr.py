"""class model:
    def __init__(self):

"""
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def svr_func(df):
  train_df = df
  #Data Preprocessing
  train_df_svr = train_df.head(len(train_df)-1) 
  days = train_df_svr.index.tolist()
  days = [[i] for i in days]
  #adj_prices = [[i] for i in adj_prices]
  adj_prices = train_df_svr['Close'].tolist()
  train_X, test_X, train_y, test_y = train_test_split(days, adj_prices, test_size=0.25)

  #SVR Models-1
  #lin_svr = SVR(kernel='linear',C=1000.0)
  #lin_svr.fit(train_X,train_y)

  #SVR Models-2
  poly_svr = SVR(kernel='poly',C=1000.0, degree=2)
  poly_svr.fit(train_X,train_y)

  #SVR Models-3
  rbf_svr = SVR(kernel='rbf',C=1000.0, gamma = 0.15)
  rbf_svr.fit(train_X,train_y)

  #performance evaluation

  print("Pure prediction section")
  print("Actual score - poly")
  print(poly_svr.score(test_X, test_y))
  print("Actual score - rbf")
  print(rbf_svr.score(test_X, test_y))

  print("------------------------------------")
  #svr_list = [lin_svr,poly_svr,rbf_svr]
  svr_list = [poly_svr,rbf_svr]
  for svr in svr_list:
    svr_predict = svr.predict(test_X)

    print('SVR {} PERFORMANCE'.format(str(svr)))
    print('r2 score: '+str(r2_score(test_y, svr_predict)))
    print('RMSE : '+str(np.sqrt(mean_squared_error(test_y, svr_predict))))
    print("Mean Absolute Error : " + str(mean_absolute_error(test_y,svr_predict)))

from sklearn import svm

def svm_func(df_):

  print("Momentum prediction - as classifier")

  length = len(df_)
  df = df_

  X = np.transpose(np.array([df['percetage_change'],df['bollinger_gap'],df['bollinger_mean'],df['bollinger_std']]))
  Y = np.array(df['momentum'])

  X_train = X[0:int(0.8*length)]
  X_test = X[int(0.8*length):]
  y_train = Y[0:int(0.8*length)]
  y_test = Y[int(0.8*length):]


  poly_svm = svm.SVC(kernel='poly')
  poly_svm.fit(X_train, y_train)
  poly_score = poly_svm.score(X_test, y_test)
  print("Prediction score for poly SVM classifier : ",poly_score)

  rbf_svm = svm.SVC(kernel='rbf')
  rbf_svm.fit(X_train, y_train)
  rbf_score = rbf_svm.score(X_test, y_test)
  print("Prediction score for RBF SVM classifier : ",rbf_score)

  #Finding the Best search
