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
	:param np.array data:
      2D array de shape (n_data, n_features).
    :param int nperseg:
      largo de ejemplos en cada ventana temporal.
    :param int noverlap:
      numero de punto que se superponen entre un segmento y el siguiente.
    :returns:
      a numpy array of size (n_windows, nperwd, n_features).
    """
	
	# obtener dimensiones del array
	n_data = data.shape[0]
	n_features = data.shape[1]
	nleap = nperseg - noverlap
	
	# determinar cantidad de ventanas a generar
	n_windows = np.floor( (n_data - nperseg)/nleap ) + 1
	n_windows = int(n_windows)
	
	# inicializar dataset
	X = np.zeros( (n_windows, nperseg, n_features) )
	
	# generar time windows
	for i in range(n_windows):
		# obtener index de la ventana
		idx_start, idx_end = i*nleap, i*nleap + nperseg
		
		# asignar datos a X
		X[i, :] = data[idx_start:idx_end]
		
	return X	
# ----------------------------------------------------------------------------

def generate_dataset_sp(df, fs=1.0, window=('tukey', 0.25), nperseg=None, 
						noverlap=None, nfft=None, detrend='constant', 
						return_onesided=True, scaling='density', 
						axis=- 1, mode='psd'):
							  
	"""
	Devuelve X y las etiquetas Y en one-hot-encoding
	"""
	
	# obtener keys del datafram
	keys = list(df.columns)

	# inicializar listas vacías
	X = list()
	Y = list()
	
	# guardar series temporales en X
	for i in keys:
		X.append( np.array( df[i] ) )

	# obtener espectrogramas y guardarlos en la lista
	for i in range(len(keys)):
		_, _, X[i] = signal.spectrogram(X[i], fs=fs, window=window, nperseg=nperseg,
										noverlap=noverlap, nfft=nfft,detrend=detrend,
										return_onesided=return_onesided,scaling=scaling,
										axis=axis, mode=mode)
		
		# corregir shape
		X[i] = X[i].transpose()

		# generar  etiquetas
		Y.append( [i]*X[i].shape[0] )
		
	# juntar todos los espectrogramas en un solo np.array
	X = np.vstack(X)
	
	# generar etiquetas
	i = len(keys)
	Y = np.reshape( np.array(Y), (-1, 1) )
	Y = to_categorical(Y, i)

	return X, Y
# ----------------------------------------------------------------------------

def normalizar_sp(X_train, X_val, X_test):
	# inicializar MinMaxScaler
	scaler = MinMaxScaler( feature_range=(0, 1) )
		
	# fit scaler con los datos de entrenamiento X_train
	scaler.fit(X_train)
		
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)
	X_test = scaler.transform(X_test)
	
	return X_train, X_val, X_test
# ----------------------------------------------------------------------------

def generate_dataset_sp2(df,train=0.7, val=0.2, normalizar=True, fs=1.0,
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
		sp_test = sp_test.transpose()

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

def generate_dataset_sc(df, nperseg=30, noverlap=15, n_features=30,
						wavelet=signal.ricker):
	"""
	A partir de un dataframe genera los conjuntos de datos X y las etiquetas Y
	en formato one-hot-encoding. Utilizando el método cwt.
	
	:param pd.dataframe df:
		dataframe que en cada columna contiene una serie temporal correspondiente
		a la medición de vibraciones de un rodamiento bajo determinadas condiciones
		de operación.
		
	:param int nperseg:
		largo de cada segmento (Ventanas temporales Funcion_Pancho)
		
	:param int noverlap:
		numero de punto que se superponen entre un segmento y el siguiente.
		
	:param int n_features:
		cantidad de elementos que tiene el np.array Scales en el método CWT.
		
	:param wavelet:
		Wavelet que se utiliza en el método CWT.
	"""
	
	# obtener keys del dataframe
	keys = list(df.columns)
	
	# inicializar listas vacías
	X = list()
	Y = list()
	
	# guardar series temporales en X
	for i in keys:
		X.append( np.array( df[i] ) )
	
	#obtener escalograma y guardarlos en listas
	widths = np.arange( 1, n_features+1 )
	
	for i in range(len(keys)):
		X[i] = signal.cwt(X[i], signal.ricker, widths)
		
		#corregir shape
		X[i] = X[i].transpose()
		
		# Generar array 3D con ventanas temporales
		X[i] = get_time_windows_3D(X[i], nperseg, noverlap)
		
		# generar etiquetas
		Y.append( [i]*X[i].shape[0] )

	
	# juntar todos los escalogramas en un solo np.array
	X = np.vstack(X)
	
	# generar etiquetas
	i = len(keys)
	Y = np.reshape( np.array(Y), (-1, 1) )
	Y = to_categorical(Y, i)

	return X, Y
# ----------------------------------------------------------------------------

def normalizar_sc(X_train, X_val, X_test):
	nperseg = X_train.shape[1]
	n_features = X_train.shape[2]
	
	# convertir a 2D
	X_train = X_train.reshape(-1, n_features)
	X_val = X_val.reshape(-1, n_features)
	X_test = X_test.reshape(-1, n_features)
	
	# inicializar MinMaxScaler
	scaler = MinMaxScaler( feature_range=(0, 1) )
	
	# fit scaler con los datos de entrenamiento X_train
	scaler.fit(X_train)
		
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)
	X_test = scaler.transform(X_test)
	
	# convertir a 3D
	X_train = X_train.reshape(-1, nperseg, n_features)
	X_val = X_val.reshape(-1, nperseg, n_features)
	X_test = X_test.reshape(-1, nperseg, n_features)
	
	return X_train, X_val, X_test
# ----------------------------------------------------------------------------

def plot_confusion_matrix(Y_true, Y_pred, target_names,
                          title='Confusion matrix',
                          cmap=None, normalize=False,
                          figsize=(5,5)):
    
    """
    given the true (Y_true) and the predicted (Y_pred) labels,
    makes the confusion matrix.
    
    :param np.array Y_true:
        the true labels of the data. (no one hot encoding).
    :param np.array Y_pred:
        the predicted labels of the data by the model. (no one hot encoding).
    :param list target_names:
        given classification classes such as [0, 1, 2] the class names,
        for example: ['high', 'medium', 'low'].
    :param str title:
        the text to display at the top of the matrix.
    :param str cmap:
        the gradient of the values displayed from matplotlib.pyplot.cm
        see http://matplotlib.org/examples/color/colormaps_reference.html
        plt.get_cmap('jet') or plt.cm.Blues.
    :param bool normalize:
        if False, plot the raw numbers, if True, plot the proportions.
        
    :reference:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        
    """
    import itertools
    
    cm = confusion_matrix(Y_true, Y_pred)
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                      verticalalignment="center",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      verticalalignment="center",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
    
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
# ----------------------------------------------------------------------------
def plot_loss_function(train_info, figsize=(5,5)):
    """
    -> None
    
    this function plots de evolution of the loss function of the model 
    during the training epochs.
    
    :param train_info:
        training history of the classification model.
        
    """
    # crear figura
    plt.figure(figsize=figsize)
    
    plt.plot(train_info.history['loss'])
    plt.plot(train_info.history['val_loss'])
    
    # caracteristicas del plot
    plt.title('Model loss')
    plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
 
# ----------------------------------------------------------------------------