# Imports

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder

# AI Libraries
import tensorflow as tf

# General libraries
import os
import re


# Functions
def load_audios(dir_name):
	audios = []
	
	datasets = map(lambda x: dir_name + '\\' + x, os.listdir(dir_name))
	
	for i in datasets:
		if 'train' in i:
			for j in os.listdir(i):
				name = os.path.basename(i + '\\' + j)
				audio = (i + '\\' + j, name.split('.')[0], 'train')
				audios.append(audio)
				
		elif 'test' in i:
			for j in os.listdir(i):
				name = os.path.basename(i + '\\' + j)
				audio = (i + '\\' + j, name.split('.')[0], 'test')
				audios.append(audio)

	return audios
				
def audio_to_spectrogram(dir_name, audios):
	if not os.path.exists(dir_name + '\\' + 'images'):
		os.makedirs('images')
		os.makedirs('images/train')
		os.makedirs('images/test')
	elif len(os.listdir(dir_name + '\\' + 'images\\train')) == 0 and len(os.listdir(dir_name + '\\' + 'images\\test')) == 0:
		
		images_path = dir_name + '\\' + 'images'
		images_dataset = list(map(lambda x: images_path + '\\' + x, os.listdir(images_path)))
		
		for content in audios:
			
			try:
				sample_rate, samples = wavfile.read(content[0])				
			except Exception as e:
				raise("File {0} generated: {1}".format(os.path.basename(content[0]), e))
				break
				
			frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
			spectrogram = 10 * np.log10(spectrogram)
			plt.pcolormesh(times, frequencies, spectrogram, shading='auto')

			name = os.path.basename(content[0])
			pattern = r'.wav'
			clean_name = re.sub(pattern, '', name)
			
			plt.axis('off')
			plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
			
			if 'train' in content[2]:
				plt.savefig(images_dataset[1] + '\\' + clean_name + '.jpg', bbox_inches='tight', pad_inches=0, dpi=100)
				plt.close()
				
			elif 'test' in content[2]:			
				plt.savefig(images_dataset[0] + '\\' + clean_name + '.jpg', bbox_inches='tight', pad_inches=0, dpi=100)
				plt.close()
	else:
		pass

def image_to_array(dir_name):
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	
	images_path = dir_name + '\\' + 'images'
	images_dataset = list(map(lambda x: images_path + '\\' + x, os.listdir(images_path)))
	
	for i in images_dataset:
		if 'train' in i:
			for j in os.listdir(i):
				name = os.path.basename(i + '\\' + j)
				X_train.append(imread(i + '\\' + j))
				y_train.append(name.split('.')[0])
				
		elif 'test' in i:
			for j in os.listdir(i):
				name = os.path.basename(i + '\\' + j)
				X_test.append(imread(i + '\\' + j))
				y_test.append(name.split('.')[0])

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_test = np.array(X_test)
	y_test = np.array(y_test)
	
	X_train = X_train / 255
	X_test = X_test / 255
	
	encoder = LabelEncoder()
	y_train = encoder.fit_transform(y_train)
	y_test = encoder.fit_transform(y_test)
	
	return X_train, y_train, X_test, y_test

# Model
def model(X_dim):
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=((X_dim[1], X_dim[2], X_dim[3]))),
		tf.keras.layers.MaxPooling2D(3, 3),

		tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
		tf.keras.layers.MaxPooling2D(2, 2),

		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
		tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))
	])
	
	return model


# Main
if __name__ == '__main__':
	current_dir = os.path.dirname(os.path.abspath(__file__))
	
	if not os.path.exists(current_dir + '\\' + 'images'):
		audios_dir = current_dir + '\\audios-full'
		
		audios = load_audios(audios_dir)
		
		audio_to_spectrogram(current_dir, audios)
	
	X_train, y_train, X_test, y_test = image_to_array(current_dir)
	
	model = model(X_train.shape)
	
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=30)

	test_loss, test_acc = model.evaluate(X_test, y_test)
	print(test_acc)

	predictions = model.predict(X_test)


