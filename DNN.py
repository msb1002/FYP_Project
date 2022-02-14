import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def dnn(df):
    #Data Pre-processing
    train_df = df

    X = train_df.drop('Close',axis=1)
    Y = train_df[['Close']]

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X[X.columns] = X_scaler.fit_transform(X[X.columns])
    X = X.fillna(X.median())

    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    Y[Y.columns] = Y_scaler.fit_transform(Y[Y.columns])
    Y = Y.fillna(Y.median())

    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.25)

    #Model
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(len(X.columns),)),
            layers.Dense(128, activation="relu", name="layer1"),
            layers.Dense(256, activation="relu", name="layer2"),
            layers.Dense(1,activation="linear", name="layer3")
        ]
    )

    model.compile(loss='mean_squared_error',optimizer='adam')
    history = model.fit(train_X,train_y,validation_data=(val_X, val_y),epochs = 200 ,batch_size = 10,verbose=True)#verbose=False if needed

    history_df = pd.DataFrame(history.history)
    plt.plot(history_df['loss'],label="training Loss");
    plt.plot(history_df['val_loss'],label="Validation Loss");
    plt.title("Simple DNN Architecture")
    plt.show()

    print(model.summary())

    dnn_predict = model.predict(test_X)

    print('SIMPLE DNN PERFORMANCE')
    print('r2 score: '+str(r2_score(test_y, dnn_predict)))
    print('RMSE : '+str(np.sqrt(mean_squared_error(test_y, dnn_predict))))
    print("Mean Absolute Error : " + str(mean_absolute_error(test_y,dnn_predict)))