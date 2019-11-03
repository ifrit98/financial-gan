
import time
import numpy as np
import pandas as pd

from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


context = mx.cpu(); model_ctx=mx.cpu()
mx.random.seed(1719)


def parser(x):
  return datetime.datetime.strptime(x, '%m/%d/%Y')


dataset_OG = pd.read_csv('data/Google_Stock_Price_Train.csv', 
                            header = 0, 
                            parse_dates=[0], 
                            date_parser=parser)
                            
dataset_OG[['Date', 'Open']].head(3)
dataset_OG.head(3)
print('There are {} number of days in the dataset.'.format(
  dataset_OG.shape[0]))

# Visualize the stock for the last several years
plt.figure()
plt.plot(dataset_OG['Date'], dataset_OG['Open'], label='Google stock')
plt.vlines(datetime.date(2016, 4, 20), 0, 800, linestyles='--', 
           colors='gray',
           label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Figure 1: Google stock price')
plt.savefig('plots/google-stock.pdf')
plt.legend()
plt.show()


num_training_days = int(dataset_OG.shape[0]*.7)
print('Number of training days: {}. Number of test days: {}.'.format(
  num_training_days, dataset_OG.shape[0]-num_training_days))


## Technical Indicators
def convert_to_numeric(ds):
  ds['Close'] = pd.to_numeric(ds['Close'], errors='coerce')
  ds['Open'] = pd.to_numeric(ds['Open'], errors='coerce')
  ds['High'] = pd.to_numeric(ds['High'], errors='coerce')
  ds['Low'] = pd.to_numeric(ds['Low'], errors='coerce')
  return ds


def get_technical_indicators(ds, attr = 'Open'):
  # 7 and 21 day moving average
  ds['ma7'] = ds[attr].rolling(window=7).mean()
  ds['ma21'] = ds[attr].rolling(window=21).mean()
  
  # Create MACD
  ds['26ema'] = ds[attr].ewm(span=26).mean()
  ds['12ema'] = ds[attr].ewm(span=12).mean()
  ds['MACD'] = (ds['12ema']-ds['26ema'])
  
  # Create Bollinger Bands
  ds['20sd'] = ds['Close'].rolling(20).std()
  ds['upper_band'] = ds['ma21'] + (ds['20sd']*2)
  ds['lower_band'] = ds['ma21'] - (ds['20sd']*2)
  
  # Create Exponential moving average
  ds['ema'] = ds[attr].ewm(com=0.5).mean()
  
  # Create Momentum
  ds['momentum'] = ds[attr]-1
  return ds


ds = convert_to_numeric(dataset_OG)
ds_ti = get_technical_indicators(ds)                                                                  
print(ds_ti.head(5))


def plot_technical_indicators(ds, last_days):
  plt.figure(figsize=(16, 10), dpi=100)
  shape_0 = ds.shape[0]
  xmacd_ = shape_0-last_days
  
  ds = ds.iloc[-last_days:, :]
  x_ = range(3, ds.shape[0])
  x_ = list(ds.index)
  
  # Plot first subplot
  plt.subplot(2, 1, 1)
  plt.plot(ds['ma7'],label='MA 7', color='g',linestyle='--')
  plt.plot(ds['Close'],label='Closing Price', color='b')
  plt.plot(ds['ma21'],label='MA 21', color='r',linestyle='--')
  plt.plot(ds['upper_band'],label='Upper Band', color='c')
  plt.plot(ds['lower_band'],label='Lower Band', color='c')
  plt.fill_between(x_, ds['lower_band'], ds['upper_band'], alpha=0.35)
  plt.title('Figure 2: Technical indicators for Google - last {} days. \
  '.format(last_days))
  plt.ylabel('USD')
  plt.legend()
  
  # Plot second subplot
  plt.subplot(2, 1, 2)
  plt.title('MACD')
  plt.plot(ds['MACD'],label='MACD', linestyle='-.')
  plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
  plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
  plt.plot(ds['momentum'],label='Momentum', color='b',linestyle='-')
  
  plt.legend()
  plt.savefig('plots/technical-indicators.pdf')
  plt.show()
  
plot_technical_indicators(ds_ti, 400)


# Do sentiment analysis here with BERT
# Bidirectional Embedding Representations from Transformers
# Pretrained BERT models are already available in MXNet/Gluon. 
# We just need to instantiated them and add two (arbitrary number)
# Dense layers, going to softmax - the score is from 0 to 1.
import bert


# Fourier transform for trend analysis
def extract_fourier(ds, attr='Open'):
  ds_FT = ds[['Date', attr]]
  close_fft = np.fft.fft(np.asarray(ds_FT[attr].tolist()))
  fft_df = pd.DataFrame({'fft':close_fft})
  fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
  fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
  
  plt.figure(figsize=(14, 7), dpi=150)
  fft_list = np.asarray(fft_df['fft'].tolist())
  for num_ in [3, 6, 9, 100]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), 
             label='Fourier transform with {} components'.format(num_))  
  plt.plot(ds_FT[attr],  label='Real')
  plt.xlabel('Days')
  plt.ylabel('USD')
  plt.title('Figure 3: Google stock prices & Fourier transforms')
  plt.legend()
  plt.savefig('plots/Fourier-Google.pdf')
  plt.show()
  
  return fft_df

fft_df = extract_fourier(ds)


# Show FFT (delta-like spike function)
from collections import deque
items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df)/2)))
plt.figure(figsize=(10, 7), dpi=100)
plt.stem(items)
plt.title('Figure 4: Components of Fourier transforms')
plt.savefig('plots/fft-google.pdf')
plt.show()


# ARIMA as a feature
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime

data_FT = ds[['Date', 'Open']]
series = data_FT['Open']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# Autocorrelation
from pandas.plotting import autocorrelation_plot
plt.figure()
plot = autocorrelation_plot(series)
plt.savefig('plots/autocorr-google.pdf')
plt.show()

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]
    
autocorrelation = pd.DataFrame({'autocorr': autocorr(series)}, dtype = np.float32)
ds_plus = pd.concat([fft_df, ds_ti, autocorrelation], axis = 1)


from pandas import read_csv
from sklearn.metrics import mean_squared_error
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# Plot the predicted (from ARIMA) and real prices
plt.figure()
plt.plot(test, label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 5: ARIMA model on GS stock')
plt.legend()
plt.savefig('plots/ARIMA-preds.pdf')
plt.show()
plt.clf()


# Aggregate all data thus-far
dataset_full = pd.concat([fft_df, ds_ti, autocorrelation], axis = 1)
dataset_full = dataset_full.drop(columns=['Date'])

print('Total dataset has {} samples, and {} features.'.format(
  dataset_full.shape[0], dataset_total_df.shape[1]))


## Feature engineering
# XGBoost feature importance
def get_feature_importance_data(data_income, attr):
  data = data_income.copy()
  y = data[attr]
  X = data.iloc[:, 1:]
  
  train_samples = int(X.shape[0] * 0.66)
  
  X_train = X.iloc[:train_samples]
  X_test = X.iloc[train_samples:]
  
  y_train = y.iloc[:train_samples]
  y_test = y.iloc[train_samples:]
  
  return (X_train, y_train), (X_test, y_test)


def test_all_features(ds):
  for i in range(len(ds.columns)):
    if ds[ds.columns[i]].dtype == complex: continue
  
    (X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(ds, ds.columns[i])
    
    
    regressor = xgb.XGBRegressor(gamma=0.0, 
                                 n_estimators=150, 
                                 base_score=0.7,
                                 colsample_bytree=1,
                                 learning_rate=0.05)
    
    xgbModel = regressor.fit(X_train_FI, y_train_FI,
                             eval_set=[(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
                             verbose=False)
    
    eval_result = regressor.evals_result()
    
    training_rounds = range(len(eval_result['validation_0']['rmse']))
    
    plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
    plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Training Vs Validation Error')
    plt.legend()
    plt.savefig('plots/XGBoost_results/' + ds.columns[i] + '_.png')
    # plt.show()
    plt.clf()
    
    fig = plt.figure()
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
    plt.title('Figure 6: Feature importance of the technical indicators.')
    plt.savefig('plots/feature_importance/' + ds.columns[i] + '_.png')
    # plt.show()
    plt.clf()
  

test_all_features(dataset_full)





# Add gelu to MXnext implementation
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
def relu(x):
    return max(x, 0)
def lrelu(x):
    return max(0.01*x, x)

# Visualize some activation functions
plt.figure(figsize=(15, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)

ranges_ = (-10, 3, .25)

plt.subplot(1, 2, 1)
plt.plot([i for i in np.arange(*ranges_)], [relu(i) for i in np.arange(*ranges_)], label='ReLU', marker='.')
plt.plot([i for i in np.arange(*ranges_)], [gelu(i) for i in np.arange(*ranges_)], label='GELU')
plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
plt.title('Figure 7: GELU as an activation function for autoencoders')
plt.ylabel('f(x) for GELU and ReLU')
plt.xlabel('x')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([i for i in np.arange(*ranges_)], [lrelu(i) for i in np.arange(*ranges_)], label='Leaky ReLU')
plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
plt.ylabel('f(x) for Leaky ReLU')
plt.xlabel('x')
plt.title('Figure 8: LeakyReLU')
plt.legend()
plt.show()
plt.clf()




## Create VAE (optional)
# TODO: Make your own VAE in tf (simplified/my_wheelhouse)

model_ctx =  mx.cpu()
class VAE(gluon.HybridBlock):
    def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784, \
                 batch_size=100, act_type='relu', **kwargs):
        self.soft_zero = 1e-10
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.output = None
        self.mu = None
        super(VAE, self).__init__(**kwargs)
        
        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix='encoder')
            
            for i in range(n_layers):
                self.encoder.add(nn.Dense(n_hidden, activation=act_type))
            self.encoder.add(nn.Dense(n_latent*2, activation=None))

            self.decoder = nn.HybridSequential(prefix='decoder')
            for i in range(n_layers):
                self.decoder.add(nn.Dense(n_hidden, activation=act_type))
            self.decoder.add(nn.Dense(n_output, activation='sigmoid'))

    def hybrid_forward(self, F, x):
        h = self.encoder(x)
        #print(h)
        mu_lv = F.split(h, axis=1, num_outputs=2)
        mu = mu_lv[0]
        lv = mu_lv[1]
        self.mu = mu

        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx)
        z = mu + F.exp(0.5*lv)*eps
        y = self.decoder(z)
        self.output = y

        KL = 0.5*F.sum(1+lv-mu*mu-F.exp(lv),axis=1)
        logloss = F.sum(x*F.log(y+self.soft_zero)+ (1-x)*F.log(1-y+self.soft_zero), axis=1)
        loss = -logloss-KL

        return loss


batch_size = 64
n_batches = VAE_data.shape[0]/batch_size
VAE_data = VAE_data.values

train_iter = mx.io.NDArrayIter(data={'data': VAE_data[:num_training_days,:-1]}, \
                               label={'label': VAE_data[:num_training_days, -1]}, batch_size = batch_size)
test_iter = mx.io.NDArrayIter(data={'data': VAE_data[num_training_days:,:-1]}, \
                              label={'label': VAE_data[num_training_days:,-1]}, batch_size = batch_size)


n_hidden = 400 # neurons in each layer
n_latent = 2 
n_layers = 3 # num of dense layers in encoder and decoder respectively
n_output = VAE_data.shape[1] - 1 

net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size, act_type='gelu')


net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .01})

## TODO: Anomaly detection with deep unsupervised learning in derivates pricing ??

print(net)


## Train VAE
n_epoch = 150
print_period = n_epoch // 10
start = time.time()

training_loss = []
validation_loss = []
for epoch in range(n_epoch):
    epoch_loss = 0
    epoch_val_loss = 0

    train_iter.reset()
    test_iter.reset()

    n_batch_train = 0
    for batch in train_iter:
        n_batch_train +=1
        data = batch.data[0].as_in_context(mx.cpu())

        with autograd.record():
            loss = net(data)
        loss.backward()
        trainer.step(data.shape[0])
        epoch_loss += nd.mean(loss).asscalar()

    n_batch_val = 0
    for batch in test_iter:
        n_batch_val +=1
        data = batch.data[0].as_in_context(mx.cpu())
        loss = net(data)
        epoch_val_loss += nd.mean(loss).asscalar()

    epoch_loss /= n_batch_train
    epoch_val_loss /= n_batch_val

    training_loss.append(epoch_loss)
    validation_loss.append(epoch_val_loss)

    """if epoch % max(print_period, 1) == 0:
        print('Epoch {}, Training loss {:.2f}, Validation loss {:.2f}'.\
              format(epoch, epoch_loss, epoch_val_loss))"""

end = time.time()
print('Training completed in {} seconds.'.format(int(end-start)))


dataset_total_df['Date'] = dataset_ex_df['Date']
vae_added_df = mx.nd.array(dataset_total_df.iloc[:, :-1].values)
print('The shape of the newly created (from the autoencoder) features is {}.'.format(vae_added_df.shape))


# Eigen portfolio with PCA
# We want the PCA to create the new components to explain 80% of the variance
pca = PCA(n_components=.8)

x_pca = StandardScaler().fit_transform(vae_added_df)

principalComponents = pca.fit_transform(x_pca)

principalComponents.n_components_

"""
 So, in order to explain 80% of the variance we need 84 (out of the 112) features. 
 This is still a lot. So, for now we will not include the autoencoder created features. 
 Let's work on creating the autoencoder architecture in which we get the output from an 
 intermediate layer (not the last one) and connect it to another Dense layer with, say, 
 30 neurons. Thus, we will 1) only extract higher level features, and 2) come up with 
 significantly fewer number of columns.

"""


