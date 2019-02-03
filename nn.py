from keras.datasets import mnist
import keras
from keras.regularizers import l2
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
#from keras.utils import plot_model


BATCH_SIZE=128
EPOCHS=10
L_RATE = 0.01
REG = 0.01
ACTIVATION = 'relu'


FILENAME = 'mlp'+str(BATCH_SIZE)



def mlp_model(shape):
	"""Create a multilayer perseptron model
	
	Args:
	    shape (tuple): Shape of input
	
	Returns:
	    Model: Model
	"""
	inp = Input(shape = shape)
	hidden = Dense(512,activation=ACTIVATION,kernel_regularizer=l2(REG))(inp)
	outp = Dense(10,activation='softmax',kernel_regularizer=l2(REG))(hidden)	
	return Model(inputs = inp, outputs = outp) 

def load_data():
	"""Load data from keras.dataset.mninst
	
	Returns:
	    (list, list): training set, test set
	"""
	return mnist.load_data()

def preprocessing(data, X_dim = 2):
	"""preprossess the inputdata from keras dataset to desired shape,
	normalizing trainingdata and reshaping labels to categorical vector. 
	
	Args:
	    data ((list,list)(list,list)): datasets from keras.dataset containing training and test sets
	    X_dim (int, optional): dimention of X sets desired for the model. 2 dimentions for Dense and 4 for Convolutional
	
	Returns:
	    ((list,list),(list,list),(list,list),(tuple)): retuns training set, validation set, test set and shape of X-data
	"""
	(x_train, y_train), (x_test, y_test) = data
	print("Shape of x_train unprocessed :", x_train.shape)
	
	if X_dim == 2:
		inp_shape = (x_train.shape[1] * x_train.shape[2],) 
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]* x_train.shape[2]).astype('float32')
		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2]).astype('float32')
	elif X_dim == 4:	
		inp_shape = (x_train.shape[1] , x_train.shape[2],1,) 
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1).astype('float32')
		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1).astype('float32')

	x_train /= 255
	x_test /= 255

	y_train = keras.utils.to_categorical(y_train,10)
	y_test = keras.utils.to_categorical(y_test,10)

	x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train, test_size = 1/12)

	return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), inp_shape

def save_model(model, name):
	"""Save model to json and weigths to HDF5 files 
	
	Args:
	    model (Model): The model to save
	    name (string): Filename of file to save to
	"""
	with open(name+'.json','w') as f:
		f.write(model.to_json())
	model.save_weights(name+'.h5')

def load_model(name):
	"""Load model from json and weigths from HDF5 files 
	
	Args:
	    name (string): Filename of file to load from
	"""
	with open(name+'.json','r') as f:
		model = model_from_json(f.read())
	model.load_weights(name+'.h5')
	return model

def confusion_matrix(model,X,y):
	"""Create confusion matrix of predictions
	
	Args:
	    model (Model): Model
	    X (list): data to predict on
	    y (list): labels
	
	Returns:
	    matrix: confusion matrix
	"""
	conf_matrix = np.zeros((10,10),dtype= int)
	predicts = model.predict(X)

	errorEx =[]
	n = 0
	for xi,yi,img in zip(predicts,y,X):
		pred = int(np.argmax(xi))
		corr = int(np.argmax(yi))
		if pred != corr and n<6:
			errorEx.append(img)
			n+=1
		conf_matrix[pred][corr]+=1
	save_error_examples(errorEx)
	return conf_matrix

def save_error_examples(errors):
	"""Create and store plot of missclassified images
	
	Args:
	    errors (array): Array of images to plot
	"""
	fig, axs = plt.subplots(nrows=2, ncols=3)
	for img, ax in zip(errors,axs.flat):
		img = img.reshape((28, 28))
		ax.imshow(img, cmap='gray')
	plt.savefig(FILENAME+'_'+'error_example'+'.png')
	plt.close()

def array_to_csv(array, model_name, array_name):
	"""Save an array to a csv file
	"""
	pd.DataFrame(array).to_csv(model_name+'_'+array_name+'.csv',header=None, index=None)





if __name__ == '__main__':

	train_set, valid_set, test_set, inp_shape = preprocessing(load_data())
	
	nn = mlp_model(inp_shape)
	nn.summary()

	nn.compile(loss='categorical_crossentropy',  optimizer=SGD(lr=L_RATE), metrics=['accuracy'])

	result = nn.fit(train_set[0],train_set[1],
	                    batch_size=BATCH_SIZE,
	                    epochs=EPOCHS,
	                    verbose=2,
	                    validation_data=valid_set)
	
	save_model(nn, FILENAME)
	#nn = load_model(FILENAME)
	#nn.compile(loss='categorical_crossentropy',  optimizer='sgd', metrics=['accuracy'])




	score = nn.evaluate(test_set[0], test_set[1], verbose=0)

	conf_matrix = confusion_matrix(nn,test_set[0],test_set[1])

	print(conf_matrix)

	print('loss:',score[0])
	print('accuracy:',score[1])

	array_to_csv(result.history['acc'], FILENAME, 'train_accuracy')
	array_to_csv(result.history['val_acc'], FILENAME, 'val_accuracy')
	array_to_csv(result.history['loss'], FILENAME, 'train_loss')
	array_to_csv(result.history['val_loss'], FILENAME, 'val_loss')
	array_to_csv(conf_matrix, FILENAME, 'conf_matrix')
	#plot_model(nn, to_file='cnn.png')
	print('plotting...')

	plt.figure(2)
	plt.plot(result.history['acc'])
	plt.plot(result.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.savefig(FILENAME+'_'+'acc_plot'+'.png')
	#plt.show()
	plt.close()

	plt.figure(3)
	plt.plot(result.history['loss'])
	plt.plot(result.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.savefig(FILENAME+'_'+'loss_plot'+'.png')
	#plt.show()
	plt.close()

	print('done')