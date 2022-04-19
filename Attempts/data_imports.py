import pandas_datareader.data as web

#Setting the index values as timestamp
def process():
  print("Enter the code of the model that you want to implement")
  code = input()
  #final_code = '^'+code
  final_code = code
  df_ = web.DataReader(final_code, 'stooq')
  #df_['timestamp'] = df_.index.astype('int64')
  df_.reset_index(drop=True, inplace=True)
  #df_.set_index('timestamp', inplace=True)

  print(f"Loading the dataset for the {code}")

  return df_

def additional_features(df):
      #SMA
  df['10_sma'] = df['Close'].rolling(window=10,min_periods=1).mean()
  df['20_sma'] = df['Close'].rolling(window=20,min_periods=1).mean()
  df['50_sma'] = df['Close'].rolling(window=50,min_periods=1).mean()
  
  #EMA
  df['10_ema'] = df['Close'].ewm(span=10,min_periods=1).mean()
  df['20_ema'] = df['Close'].ewm(span=20,min_periods=1).mean()
  df['50_ema'] = df['Close'].ewm(span=50,min_periods=1).mean()

  #Bollinger
  df['bollinger_mean'] = df['Close'].rolling(20, min_periods=1).mean()
  df['bollinger_std'] = df['Close'].rolling(20, min_periods=1).std()
  df['BOL_UP'] = df['bollinger_mean'] + (2 * df['bollinger_std'])
  df['BOL_DOWN'] = df['bollinger_mean'] - (2 * df['bollinger_std'])
  df["bollinger_gap"] = df["BOL_UP"]-df['BOL_DOWN']
  df.dropna(inplace=True) #Not sure if I can change?

  price_change =  df['Close'].pct_change().fillna(0)
  df['percetage_change'] = price_change

  momentum = [1,1]
  for i in range(2,len(df)):
    momentum.append(1 if df['Close'][i] > df['Close'][i-1] else -1)
  df['momentum'] = momentum