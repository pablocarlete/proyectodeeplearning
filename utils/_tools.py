import pandas as pd
import numpy as np
from scipy import signal
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import confusion_matrix

from scipy.stats import kurtosis
from scipy.stats import skew

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

def get_time_windows4(data, nperwd, noverlap):
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
      a numpy array of size (n_windows, nperwd, largo).
    """
	# obtener np.array de la serie de datos
	x = data.values
	# obtener np.array de la serie de datos
	n_data = x.shape[0]
	largo = x.shape[1]
	nleap =nperwd-noverlap
	# determinar cantidad de ventanas a generar
	n_windows = np.floor( (n_data - nperwd)/nleap ) + 1
	n_windows = int(n_windows)
	
	# inicializar dataset
	X = np.zeros( (n_windows, nperwd,largo) )
	# generar time windows
	for i in range(n_windows):
		# obtener index de la ventana
		idx_start, idx_end = i*nleap, i*nleap + nperwd
		# asignar datos a X
		X[i, :] = x[idx_start:idx_end]
	
	return X
	
# ----------------------------------------------------------------------------

def generate_dataset_sc(df, nperseg, noverlap, train=0.7, val=0.2, scales_len=30, wavelet='gaus1', sampling_period=1.0, normalizar=True):
	
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
	df_train, df_val, df_test = split_dataframe(df)
	
	# inicializar listas vacías
	sc_train = list()
	sc_val = list()
	sc_test = list()
	
	Y_train = list()
	Y_val = list()
	Y_test = list()
	
	#obtener escalograma de cada weá
	for i in range(len(keys)):
		coef_train, _ = pywt.cwt(np.array(df_train[keys[i]]),
								np.arange(1,scales_len+1),wavelet,
								sampling_period=sampling_period)
								
		coef_val, _ = pywt.cwt(np.array(df_val[keys[i]]),
								np.arange(1,scales_len+1),wavelet,
								sampling_period=sampling_period)
								
		coef_test, _ = pywt.cwt(np.array(df_test[keys[i]]),
								np.arange(1,scales_len+1),wavelet,
								sampling_period=sampling_period)
		
		#corregir shape
		coef_train = coef_train.transpose()
		coef_val = coef_val.transpose()
		coef_test = coef_test.transpose()
		
		# Generar array 3D con ventanas temporales (FUNCION PANCHO)
		coef_train = get_time_windows4(coef_train, nperseg, noverlap)
		coef_val = get_time_windows4(coef_val, nperseg, noverlap)
		coef_test = get_time_windows4(coef_test, nperseg, noverlap)
		
		# guardar escalogramas en la lista
		sc_train.append(coef_train)
		sc_val.append(coef_val)
		sc_test.append(coef_test)
		
		# generar etiquetas
		Y_train.append([i]*coef_train.shape[0])
		Y_val.append([i]*coef_val.shape[0])
		Y_test.append([i]*coef_test.shape[0])
		
	# juntar todos los escalogramas en un solo np.array
	X_train = np.vstack(sc_train)
	X_val = np.vstack(sc_val)
	X_test = np.vstack(sc_test)
	
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
 
