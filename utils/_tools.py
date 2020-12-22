import pandas as pd
import numpy as np


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