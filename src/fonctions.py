import os.path
import numpy as np
import shutil
import random
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.preprocessing import image
from urllib.parse import urlparse
from skimage.transform import resize
from keras.models import load_model


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

def list_index(directory):
    """
    Fonction permettant de lister les index des catégories dans le directory
    """

    listdir = os.listdir(directory + "/test")
    list_index = []
    for obj in listdir:
        list_index.append(listdir.index(obj))
    return list_index

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

def list_folder(directory):
    """
    Fonction permettant de lister le nom des dossiers (est utile juste une fois pour la fonction "clean_lite")
    """



    nom_dossier = os.listdir(directory + "/test")
    return nom_dossier

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_model(model, image, directory):
    """
    Fonction qui nous permet de tester notre modèle
    """

    listindex = list_index(directory)

    model = load_model(model)
    print("\nModèle chargé!")

    test_image = plt.imread(image)

    resized_image = resize(test_image, (32, 32, 3))
    plt.imshow(resized_image)
    predictions = model.predict(np.array([resized_image]))
    predictions
    x = predictions
    classification = list_folder(directory)

    plt.show()

    print("\nRoulement de tambour...\nPrédictions :\n")

    for i in range(len(listindex)):
        for j in range(len(listindex)):
            if x[0][listindex[i]] > x[0][listindex[j]]:
                temp = listindex[i]
                listindex[i] = listindex[j]
                listindex[j] = temp

    # Affiche les étiquettes triées dans l'ordre de la probabilité de la plus élevée à la plus faible
    # print(list_index)

    i=0
    for i in range(len(listindex)):
        print(classification[listindex[i]], ':', round(predictions[0][listindex[i]] * 100, 2), '%')
    print("\n")


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------


def training_model(kernel, stride, epochs, train_generator, test_generator, model_name):
    
    MODELS_ROOT = "saved_models/"

    model = Sequential()

    model.add(Conv2D(50, kernel_size=(kernel,kernel), strides=(stride,stride), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(125, kernel_size=(kernel, kernel), strides=(stride,stride), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    entrainement = model.fit_generator(train_generator, epochs = epochs, validation_data = test_generator)
    score = model.evaluate(train_generator, verbose=0)
    model.save(MODELS_ROOT + model_name)

    with mlflow.start_run():
            mlflow.log_param("epoch", epochs)
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("stride", stride)
            mlflow.log_metric("accuracy", score[1])
            mlflow.log_metric("loss", score[0])
            
            mlflow.keras.log_model(model, "models")


