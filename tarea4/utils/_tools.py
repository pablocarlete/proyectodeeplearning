import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc


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
def generate_ROC(X, Y, autoencoder):
  """
  -> None

  genera y gráfica la curva Receiver Operating Characteristic sobre el detector
  de anomalías dado por el modelo autoencoder entregado.

  :param np.array X:
    datos a clasificar mediante el detector de anomalías.
  :param np.array Y:
    etiquetas reales de los datos X.
  :param keras.model autoencoder:
    modelo a partir del cual se construye el detector de anomalías.

  :returns:
    gráfico ROC.
  """

  # obtener indices nominal_idx de samples nominales
  # ** esto es específico para este caso de estudio **
  RUL = Y_test.flatten() 
  nominal_idx = np.where( RUL == 1 )[0]
  degraded_idx = np.where( RUL == 0 )[0]

  # segementar samples nominales y degradados
  X_nominal = X[nominal_idx, :]
  X_degraded = X[degraded_idx, :]

  # obtener reconstrucciones mediante el autoencoder
  AE_nominal = autoencoder(X_nominal)
  AE_degraded = autoencoder(X_degraded)

  # obtener rmse de reconstrucciones
  rmse_nominal = np.sqrt( np.mean( np.power(X_nominal - AE_nominal, 2 ), axis=1) )
  rmse_degraded = np.sqrt( np.mean( np.power(X_degraded - AE_degraded, 2 ), axis=1) )

  # ---
  # generar curva ROC
  min_rmse = np.min( np.hstack([rmse_nominal, rmse_degraded]), axis=None )
  max_rmse = np.max( np.hstack([rmse_nominal, rmse_degraded]), axis=None )
  threshold = np.linspace(min_rmse, max_rmse, 1000)


  S, R = [], []
  # para cada umbral en el rango threshold
  for t in threshold:
    # obtener sensibilidad (True Positives/All Positives)
    anomalies = np.where(rmse_degraded > t)[0]
    sensibilidad = anomalies.size/degraded_idx.size

    # obtener ratio (False Positives/All Negatives)
    anomalies = np.where(rmse_nominal > t)[0]
    ratio = anomalies.size/nominal_idx.size

    # registrar valores en listas
    S.append(sensibilidad)
    R.append(ratio)

  # print AUC
  area = auc(R, S)
  print('AUC: {:1.4f}'.format(area))

  # visualizar
  plt.figure( figsize=(12, 4) )
  plt.plot(R, S)

  plt.plot([0, 1], [0, 1], 'r')
  plt.xlabel('1 - Especificidad')
  plt.ylabel('Sensibilidad')
  plt.grid(True)
  plt.show()