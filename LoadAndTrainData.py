get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
# Scikit-learn imports.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
# TensorFlow and Keras imports.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical
#Read in the cleansed and collected data
df_sig_rets = pd.read_pickle('./stockdata.pkl')
# List of all unique stock-tickers in the dataset.
tickers = df_sig_rets.reset_index()['Ticker'].unique()
#Split the tickers using a simple 80/20 rule
tickers_train, tickers_test = train_test_split(tickers, train_size=0.8)
#Create train and test dataframe
df_train = df_sig_rets.loc[tickers_train]
df_test = df_sig_rets.loc[tickers_test]
# DataFrames with signals for training
X_train = df_train.drop(columns=['Total Log Return 1-3 Years'])
X_test = df_test.drop(columns=['Total Log Return 1-3 Years'])
# DataFrames with returns
y_train = df_train['Total Log Return 1-3 Years']
y_test = df_test['Total Log Return 1-3 Years']

# Scale and fit the signals
signal_scaler = StandardScaler()
signal_scaler.fit(X_train)
# Scale the training data
array = signal_scaler.transform(X_train)
df_scaled = pd.DataFrame(data=array,columns=X_train.columns,index=X_train.index)
X_train = df_scaled
# Scale the test data
array = signal_scaler.transform(X_test)
df_scaled = pd.DataFrame(data=array,columns=X_test.columns,index=X_test.index)
X_test = df_scaled

# Create a new Keras model.
model_regr = Sequential()
# input layer of Neural Network
num_signals = X_train.shape[1]
model_regr.add(InputLayer(input_shape=(num_signals,)))
# Layers of the network
#model_regr.add(Dense(512, activation='relu'))
#model_regr.add(Dense(256, activation='relu'))
#model_regr.add(Dense(128, activation='relu'))
#model_regr.add(Dense(64, activation='relu'))
#model_regr.add(Dense(32, activation='relu'))
#model_regr.add(Dense(16, activation='relu'))
#model_regr.add(Dense(8, activation='relu'))
model_regr.add(Dense(24, activation='relu'))
#model_regr.add(Dense(16, activation='relu'))
#model_regr.add(Dense(8, activation='relu'))
# Output of the Neural Network.
model_regr.add(Dense(1))
# Compile the model
model_regr.compile(loss='mse', metrics=['mae'],optimizer=Adam(lr=0.001))
model_regr.summary()

#Arguments for Keras fit()
fit_args = {
        'batch_size': 8192,
        #num of iterations
        'epochs': 50,
        #80/20 rule for validation
        'validation_split': 0.2,
        # Show status
        'verbose': 0,
    }

get_ipython().run_cell_magic('time', '', 'history_regr = model_regr.fit(x=X_train.values,y=y_train.values, **fit_args)')
get_ipython().run_cell_magic('time', '', 'y_train_pred = model_regr.predict(X_train.values)\ny_test_pred = model_regr.predict(X_test.values)')

print('R^2 Value for trained data: ',r2_score(y_true=y_train, y_pred=y_train_pred))
print('R^2 Value for Test data: ',r2_score(y_true=y_test, y_pred=y_test_pred))

# Column-name 
TOTAL_RETURN_PRED = 'Total log Return Predicted'
# Formating the data to be plotted
df_y_train = pd.DataFrame(y_train)
df_y_train[TOTAL_RETURN_PRED] = y_train_pred
df_y_test = pd.DataFrame(y_test)
df_y_test[TOTAL_RETURN_PRED] = y_test_pred

#Creates a dataframe tickers and their r2 value
df_test_r2 = pd.DataFrame(columns = ['Ticker','r^2'])
for t in tickers_test:
    temp_df = df_y_test.loc[t]
    #add the ticker and r^2 value to the df
    r2 = r2_score(temp_df['Total Log Return 1-3 Years'],temp_df['Total log Return Predicted'])
    new_row = {'Ticker':t,'r^2':r2}
    df_test_r2 = df_test_r2.append(new_row,ignore_index=True)
df_test_r2 = df_test_r2.sort_values(by=['r^2'],ascending=False).reset_index(drop=True)
#Same as above but for trained data
df_train_r2 = pd.DataFrame(columns = ['Ticker','r^2'])
for t in tickers_train:
    temp_df = df_y_train.loc[t]
    #add the ticker and r^2 value to the df
    r2 = r2_score(temp_df['Total Log Return 1-3 Years'],temp_df['Total log Return Predicted'])
    new_row = {'Ticker':t,'r^2':r2}
    df_train_r2 = df_train_r2.append(new_row,ignore_index=True)
df_train_r2 = df_train_r2.sort_values(by=['r^2'],ascending=False).reset_index(drop=True)

def plotTestStock(userTicker):
    try:
        index = int(np.where(tickers_test == userTicker)[0])
    except:
        print('Stock ticker ',userTicker,' not found :(')
        return
    #index = int(np.where(tickers_test == userTicker)[0])
    ticker = tickers_test[index]
    title = "Predicted log stock return from test set: " + ticker 
    #negitive used to hide output
    _= df_y_test.loc[ticker].plot(title=title,  ylabel='Total log return')

def plotTrainStock(userTicker):
    try:
        index = int(np.where(tickers_train == userTicker)[0])
    except:
        print('Stock ticker ',userTicker,' not found :(')
        return
    #title = "Predicted stock return from trained set " + ticker 
    ticker = tickers_train[index]
    title = "Predicted log stock return from trained set: " + ticker 
    _= df_y_train.loc[ticker].plot(title=title,  ylabel='Total log return')

#List of training tickers
print('List of training tickers: ',tickers_train)
#List of testing tickers
print('List of testing tickers: ',tickers_test)

plotTestStock('FB')
plotTrainStock('ELY')

#Get the test stock with the highest r2 value
best = df_test_r2[df_test_r2['r^2']==df_test_r2['r^2'].max()]
#Top 10 best predicted
for i in range(0,10):
    plotTestStock(df_test_r2['Ticker'].iloc[i])
#Worst predicted
plotTestStock(df_test_r2['Ticker'].iloc[-1])
#Best train predicted
plotTrainStock(df_train_r2['Ticker'].iloc[0])
#Worst train predicted
plotTrainStock(df_train_r2['Ticker'].iloc[-20])
#median test stock
median = int(df_test_r2.shape[0]/2)
plotTestStock(df_test_r2['Ticker'].iloc[median])