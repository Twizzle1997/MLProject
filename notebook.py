#!/usr/bin/env python
# coding: utf-8

# Import des librairies
from keras.preprocessing.image import ImageDataGenerator
from src import fonctions
import sys

# Paramétrage de l'application
DATA_ROOT = "data/"
MODELS_ROOT = "saved_models/"
TRAINING_PATH = DATA_ROOT + "dataset/train"
TESTING_PATH = DATA_ROOT + "dataset/test"
DATASET = DATA_ROOT + "dataset"

img_size = (32,32)

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(directory = TRAINING_PATH, target_size = img_size, class_mode = "categorical")
test_generator = test_datagen.flow_from_directory( directory = TESTING_PATH, target_size = img_size, class_mode = "categorical")


# Paramétrage du modèle
kernel = int(sys.argv[1]) if len(sys.argv) > 1 else 3 #default:3
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1 #default:1
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15 #default:15

#Entraînement du modèle et enregistrement mlflow
fonctions.training_model(kernel, stride, epochs, train_generator, test_generator, "model_ML.h5")


# Test du modèle
fonctions.test_model("saved_models/model_ML.h5","src/images/loup_qui_tue.jpg", DATASET)

