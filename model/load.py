import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import get_session, set_session
from tensorflow.keras import backend as K

import tensorflow as tf


def init(): 
	json_file = open('yourmodel.json','r')

	sess=K.get_session()
	graph = tf.get_default_graph()
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	
	loaded_model.load_weights("yourweights.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	

	return loaded_model,graph,sess