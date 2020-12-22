import pandas as pd
import numpy as np


def split_dataframe(df, train=0.7, val=0.2):
	"""
	divide un dataframe en tres partes.
	
	:param pd.dataframe df:
		panda dataframe
	
	:param float train:
		proporción de datos de entrenamiento
		
	:param float val:
		proporción de datos de validación
		
	return:
	df_train, df_val, df_test
	"""
	indx_1 = int( train*df.shape[0])
	indx_2 = int( (train+val)*df.shape[0])
	
	df_train, df_val, df_test = df.iloc[:indx1, :], df.iloc[indx1:indx2, :], df.iloc[indx2:, :]
	
	return df_train, df_val, df_test