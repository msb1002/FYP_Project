import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import os.path as osp
import torch.utils.data as data
import torch.nn.functional as F

from tqdm import tqdm
import time, matplotlib, torch, os
from statsmodels.tsa.arima_model import ARIMA
from torch.utils.tensorboard import SummaryWriter

import os
from os import walk

import plotly
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#For DNN / CNN+LSTM
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Flatten, Dense

#For SVR
from sklearn.svm import SVR

#For SVM
from sklearn import svm
from sklearn.svm import SVC

#For ARIMA
import math
import warnings
warnings.filterwarnings('ignore')
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
#from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

#For CNN - LSTM
#import keras_tuner as kt
from keras.models import Sequential
from keras.layers import InputLayer, ConvLSTM1D, LSTM, Flatten, RepeatVector, Dense, TimeDistributed
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import keras.backend as K
#from keras.losses import Huber

print("All Iibraries Imported!")