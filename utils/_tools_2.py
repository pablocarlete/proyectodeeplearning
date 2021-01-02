import pandas as pd
import numpy as np
from scipy import signal
import pywt

import matplotlib.pyplot as plt
from matplotlib import cm

from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------------
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
		pandas dataframe df_train, df_val, df_test
	"""
	
	indx_1 = int( train*df.shape[0])
	indx_2 = int( (train+val)*df.shape[0])
	
	df_train, df_val, df_test = (df.iloc[:indx_1, :],
							df.iloc[indx_1:indx_2, :], df.iloc[indx_2:, :])
							
	if dis:
		print( 'train split: {:d} data points'.format(df_train.shape[0]) )
		print( 'validation split: {:d} data points'.format(df_val.shape[0]) )
		print( 'test split: {:d} data points'.format(df_test.shape[0]) )
		
	return df_train, df_val, df_test

# ----------------------------------------------------------------------------

def get_time_windows_3D(data, nperseg, noverlap):
	"""
	-> np.array
	generates a numpy array of time windows, of length nperwd, extracted
	from data.
	:param pd.Series x:
      time series of measurement values.
    :param int nperwd:
      largo de ejemplos en cada ventana temporal.
    :param int nleap:
      numero de punto que se superponen entre un segmento y el siguiente.
    :returns:
      a numpy array of size (n_windows, nperwd, scales_len).
    """
	
	# obtener dimensiones del array
	n_data = data.shape[0]
	scales_len = data.shape[1]
	nleap = nperseg - noverlap
	
	# determinar cantidad de ventanas a generar
	n_windows = np.floor( (n_data - nperseg)/nleap ) + 1
	n_windows = int(n_windows)
	
	# inicializar dataset
	X = np.zeros( (n_windows, nperseg, scales_len) )
	
	# generar time windows
	for i in range(n_windows):
		# obtener index de la ventana
		idx_start, idx_end = i*nleap, i*nleap + nperseg
		
		# asignar datos a X
		X[i, :, :] = data[idx_start:idx_end, :]
		
	return X
	
# ----------------------------------------------------------------------------

def generate_dataset_sp(df,train=0.7, val=0.2, normalizar=True, fs=1.0,
						window=('tukey', 0.25), nperseg=None, noverlap=None,
						nfft=None, detrend='constant', return_onesided=True,
						scaling='density', axis=- 1, mode='psd'):
							  
	"""
	función chora
	"""
  
	# obtener keys del datafram
	keys = list(df.columns)

	# seperar dataframe en train, validation y test
	df_train, df_val, df_test = split_dataframe(df, train=train, val=val)

	# inicializar listas vacías
	train = list()
	val = list()
	test = list()

	Y_train = list()
	Y_val = list()
	Y_test = list()

	# obtener espectrogramas y guardarlos en la lista
	for i in range(len(keys)):
		_, _, sp_train = signal.spectrogram(df_train[keys[i]], fs=fs, window=window,
								nperseg=nperseg, noverlap=noverlap, nfft=nfft,
								detrend=detrend, return_onesided=return_onesided,
								scaling=scaling, axis=axis, mode=mode)
								
		_, _, sp_val = signal.spectrogram(df_val[keys[i]], fs=fs, window=window,
								nperseg=nperseg, noverlap=noverlap, nfft=nfft,
								detrend=detrend, return_onesided=return_onesided,
								scaling=scaling, axis=axis, mode=mode)
								
		_, _, sp_test = signal.spectrogram(df_test[keys[i]], fs=fs, window=window,
								nperseg=nperseg, noverlap=noverlap, nfft=nfft,
								detrend=detrend, return_onesided=return_onesided,
								scaling=scaling, axis=axis, mode=mode)
								
		# corregir shape del espectrograma
		sp_train = sp_train.transpose()
		sp_val = sp_val.transpose()
		sp_test = spectrogram_test.transpose()

		# guardar espectrograma en la lista
		train.append(sp_train)
		val.append(sp_val)
		test.append(sp_test)

		# generar  etiquetas
		Y_train.append( [i]*sp_train.shape[0] )
		Y_val.append( [i]*sp_val.shape[0] )
		Y_test.append( [i]*sp_test.shape[0] )
		
	# juntar todos los espectrogramas en un solo np.array
	X_train = np.vstack(train)
	X_val = np.vstack(val)
	X_test = np.vstack(test)
	
	# normalizar
	if normalizar:
		# inicializar MinMaxScaler
		scaler = MinMaxScaler( feature_range=(0, 1) )
		
		# fit scaler con los datos de entrenamiento X_train
		scaler.fit(X_train)
		
		X_train = scaler.transform(X_train)
		X_val = scaler.transform(X_val)
		X_test = scaler.transform(X_test)
		
	# generar etiquetas
	i = len(keys)
	
	Y_train = np.reshape( np.array(Y_train), (-1, 1) )
	Y_train = to_categorical(Y_train, i)
	
	Y_val = np.reshape( np.array(Y_val), (-1, 1) )
	Y_val = to_categorical(Y_val, i)

	Y_test = np.reshape( np.array(Y_test), (-1, 1) )
	Y_test = to_categorical(Y_test, i)

	return X_train, X_val, X_test, Y_train, Y_val, Y_test
	
# ----------------------------------------------------------------------------

def generate_dataset_sc(df, nperseg, noverlap, train=0.7, val=0.2,
				scales_len=30, wavelet='gaus1', sampling_period=1.0,
				method='fft', normalizar=True):
	
	"""
	A partir de un dataframe genera los conjuntos de datos X_train, X_val,
	X_test y las etiquetas correspondientes Y_train, Y_val e Y_test. Utilizando
	el método cwt.
	
	:param pd.dataframe df:
		dataframe que en cada columna contiene una serie temporal correspondiente
		a la medición de vibraciones de un rodamiento bajo determinadas condiciones
		de operación.
		
	:param float train:
		fracción del dataframe que será utilizado para generar datos de entrenamiento.
		
	:param float val:
		fracción del dataframe que será utilizado para generar datos de validación.
		
	:param int nperseg:
		largo de cada segmento (Ventanas temporales Funcion_Pancho)
		
	:param int noverlap:
		numero de punto que se superponen entre un segmento y el siguiente.
		
	:param int scales_len:
		cantidad de elementos que tiene el np.array Scales en el método CWT.
		
	:param str wavelet:
		Wavelet que se utiliza en el método CWT.
		
	:param float sampling_period:
		periodo de muestreo utilizado en la serie temporal.
	:param boolean normalizar:
		si es True los datos X_train, X_val y X_test son normalizados entre
		(0,1) respecto al conjunto X_train, utilizando el método MinMaxScaler.
	"""
	
	# obtener keys del dataframe
	keys = list(df.columns)
	
	# separar dataframe en train, validation y test
	df_train, df_val, df_test = split_dataframe(df, train=train, val=val)
	
	# inicializar listas vacías
	train = list()
	val = list()
	test = list()
	
	Y_train = list()
	Y_val = list()
	Y_test = list()
	
	scales = np.arrange( 1, scales_len+1 )
	
	#obtener escalograma y guardarlos en listas
	for i in range(len(keys)):
		sc_train, _ = pywt.cwt(np.array(df_train[keys[i]]), scales, wavelet,
								sampling_period=sampling_period, method=method)
								
		sc_val, _ = pywt.cwt(np.array(df_val[keys[i]]), scales, wavelet,
								sampling_period=sampling_period, method=method)
								
		sc_test, _ = pywt.cwt(np.array(df_test[keys[i]]), scales, wavelet,
								sampling_period=sampling_period, method=method)
		
		#corregir shape
		sc_train = sc_train.transpose()
		sc_val = sc_val.transpose()
		sc_test = sc_test.transpose()
		
		# Generar array 3D con ventanas temporales
		sc_train = get_time_windows_3D(sc_train, nperseg, noverlap)
		sc_val = get_time_windows_3D(sc_val, nperseg, noverlap)
		sc_test = get_time_windows_3D(sc_test, nperseg, noverlap)
		
		# guardar escalogramas en la lista
		train.append(sc_train)
		val.append(sc_val)
		test.append(sc_test)
		
		# generar etiquetas
		Y_train.append( [i]*sc_train.shape[0] )
		Y_val.append( [i]*sc_val.shape[0] )
		Y_test.append( [i]*sc_test.shape[0] )
		
	# juntar todos los escalogramas en un solo np.array
	X_train = np.vstack(sc_train)
	X_val = np.vstack(sc_val)
	X_test = np.vstack(sc_test)
	
	if normalizar:
		# convertir arrays en 2D
		X_train = X_train.reshape(-1, scales_len)
		X_val = X_val.reshape(-1, scales_len)
		X_test = X_test.reshape(-1, scales_len)
		
		# inicializar MinMaxScaler
		scaler = MinMaxScaler( feature_range=(0, 1) )
	
		# fit scaler con los datos de entrenamiento X_train
		scaler.fit(X_train)
		
		X_train = scaler.transform(X_train)
		X_val = scaler.transform(X_val)
		X_test = scaler.transform(X_test)
	
	# dejar shape apta para red conv
	X_train = X_train.reshape(-1, nperseg, scales_len, 1)
	X_val = X_val.reshape(-1, nperseg, scales_len, 1)
	X_test = X_test.reshape(-1, nperseg, scales_len, 1)
	
	# generar etiquetas
	i = len(keys)

	Y_train = np.reshape( np.array(Y_train), (-1, 1) )
	Y_train = to_categorical(Y_train, i)

	Y_val = np.reshape( np.array(Y_val), (-1, 1) )
	Y_val = to_categorical(Y_val, i)

	Y_test = np.reshape( np.array(Y_test), (-1, 1) )
	Y_test = to_categorical(Y_test, i)

	return X_train, X_val, X_test, Y_train, Y_val, Y_test
# ----------------------------------------------------------------------------