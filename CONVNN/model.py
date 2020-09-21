import numpy as np
import sys
from time import time
from layer_types import *
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='cuda')

class Model:
	def __init__(self, l_layer, epochs = 1000, learning_rate = 0.01):
		self.l_layer = l_layer
		self.epochs = epochs 
		self.learning_rate = learning_rate
		self.net_layers = self.get_layers()

	def get_layers(self):
		net_layers = []
		layer = self.l_layer
		while 'p_layer' in layer.__init__.__code__.co_varnames:
			net_layers.insert(0, layer)
			layer = layer.p_layer
		return net_layers
	def train(self, x, y):
		if x.ndim != 4:
			print('the training inputs must have 4 dim (m, h, w, nc)')
			sys.exit()
		if x.shape[0] != len(y):
			print('The number of inputs should match the outputs')
			sys.exit()
		net_pred = []
		net_error = 0
		for epoch in range(self.epochs):
			print('epoch : {}/{} \n'.format(epoch, self.epochs))
			for s in range(x.shape[0]):
				self.feed(x[s,:])
				try:
					pred_y = np.where(np.max(self.l_layer.o_layer) == self.l_layer.o_layer)[0][0]
				except:
					print('Out of range')
				net_pred.append(pred_y)
				net_error = net_error + abs(pred_y - y[s])
			self.update_params(net_error)
	def feed(self, s):
		lo_layer = s
		for l in self.net_layers:
			if type(l) is Conv:
				time1 = time()
				print(lo_layer.shape)
				l.convolution(lo_layer)
				time2 = time()
				print(time2 - time1)
			elif type(l) is Dense:
				l.dense_layer(lo_layer)
			elif type(l) is Pooling:
				if l.pool == 'max':
					l.pool(lo_layer, 'max')
				if l.pool == 'average':
					l.pool(lo_layer, 'average')
			elif type(l) is Relu:
				l.relu_layer(lo_layer)
			elif type(l) is Sigmoid:
				l.sigmoid_layer(lo_layer)
			elif type(l) is Flattern:
				l.flatten(lo_layer)
			elif type(l) is Inp:
				pass
			else:
				print('this layer type is not supported by the network')
				sys.exit()
			lo_layer = l.o_layer
		return self.net_layers[-1].o_layer
	def update_params(self, net_error):
		for l in self.net_layers:
			if 'trained_params' in vars(l).keys():
				l.trained_params = l.trained_params - net_error*self.learning_rate*l.trained_params
	def predict(self, x):
		if x.ndim != 4:
			print('you should enter a 4-d data')
			sys.exit()
		preds = []
		for s in x:
			probs = self.feed(s)
			pred_y = np.where(np.max(probs) == probs)[0][0]
			preds.append(pred_y)
		return preds
	def summary(self):
		print('############## ARCHITECTURE ############## ')

		for l in self.net_layers:
			print(type(l))

		print('##########################################')





