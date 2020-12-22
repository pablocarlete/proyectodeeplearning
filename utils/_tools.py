import pandas as pd
import numpy as np
from scipy import signal
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

#---------------------------------------------------------------------------------------------------
def split_dataframe(df, train=0.7, val=0.2, dis=False):
	"""
	divide un dataframe en tres partes.
	
	:param pd.dataframe df:
		panda dataframe
	
	:param float train:
		proporción de datos de entrenamiento
		
	:param float val:
		proporción de datos de validación
		
	:param boolean dis:
		Si es True se imprime la cantidad de datos en cada dataframe
		
	return:
	df_train, df_val, df_test
	"""
	indx_1 = int( train*df.shape[0])
	indx_2 = int( (train+val)*df.shape[0])
	
	df_train, df_val, df_test = df.iloc[:indx_1, :], df.iloc[indx_1:indx_2, :], df.iloc[indx_2:, :]
	
	if dis:
		print( 'train split: {:d} data points'.format(df_train.shape[0]) )
		print( 'validation split: {:d} data points'.format(df_val.shape[0]) )
		print( 'test split: {:d} data points'.format(df_test.shape[0]) )
	
	return df_train, df_val, df_test
	
def generate_data_spectrogram(df,train=0.7, val=0.2, normalizar=True, fs=1.0,
                              window=('tukey', 0.25), nperseg=None, 
                              noverlap=None, nfft=None, detrend='constant', 
                              return_onesided=True, scaling='density', 
                              axis=- 1, mode='psd'):
							  
	"""
	función chora

	"""
  
	# obtener keys del datafram
	keys = list(df.columns)

	# seperar dataframe en train, validation y test
	df_train, df_val, df_test = split_dataframe(df, train=train, val=val)

	# inicializar lista vacía
	train = list()
	val = list()
	test = list()

	Y_train = list()
	Y_val = list()
	Y_test = list()

	# obtener espectrogramas y guardarlos en la lista
	for i in range(len(keys)):
		_, _, spectrogram_train = signal.spectrogram(df_train[keys[i]], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis, mode=mode)
    
		_, _, spectrogram_val = signal.spectrogram(df_val[keys[i]], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis, mode=mode)
    
		_, _, spectrogram_test = signal.spectrogram(df_test[keys[i]], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided,scaling=scaling, axis=axis, mode=mode)
		
		# Corregir shape del spectrograma
		spectrogram_train = spectrogram_train.transpose()
		spectrogram_val = spectrogram_val.transpose()
		spectrogram_test = spectrogram_test.transpose()

		# Guardar espectrograma en la lista
		train.append(spectrogram_train)
		val.append(spectrogram_val)
		test.append(spectrogram_test)

		# generar  etiquetas
		Y_train.append([i]*spectrogram_train.shape[0])
		Y_val.append([i]*spectrogram_val.shape[0])
		Y_test.append([i]*spectrogram_test.shape[0])

	# juntar todos los espectrogramas en un solo np.array
	X_train = np.vstack(train)
	X_val = np.vstack(val)
	X_test = np.vstack(test)
	
	# inicializar MinMaxScaler
	scaler = MinMaxScaler( feature_range=(0, 1) )
	
	# fit scaler con los datos de entrenamiento X_train
	scaler.fit(X_train)
	
	if normalizar:
		# transformar o escalar los datos del resto de los sets
		X_train = scaler.transform(X_train)
		X_val = scaler.transform(X_val)
		X_val = scaler.transform(X_val)

	# generar etiquetas
	i = len(keys)

	Y_train = np.reshape( np.array(Y_train), (-1, 1) )
	Y_train = to_categorical(Y_train, i)

	Y_val = np.reshape( np.array(Y_val), (-1, 1) )
	Y_val = to_categorical(Y_val, i)

	Y_test = np.reshape( np.array(Y_test), (-1, 1) )
	Y_test = to_categorical(Y_test, i)

	return X_train, X_val, X_test, Y_train, Y_val, Y_test
	
#------------------------------------------------
