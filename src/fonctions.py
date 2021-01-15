import os.path
import numpy as np
import shutil
import random
import matplotlib.pyplot as plt
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

