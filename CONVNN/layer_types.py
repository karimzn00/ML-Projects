import numpy as np
from activation_functions import sigmoid, relu
import sys


class Inp:
	def __init__(self, inp_shape):
		if len(inp_shape) < 2 or len(inp_shape) <= 0:
			print('you should inter a shape with at least 2 dimensions')
			sys.exit()
		elif len(inp_shape) == 2:
			inp_shape = (inp_shape[0], inp_shape[1], 1)
		self.inp_shape = inp_shape
		self.layero_size = inp_shape
		



class Conv:
	def __init__(self, filters, kernel_size, p_layer, activation_):
		if filters <= 0:
			print('number of filter should be a positive number')
			sys.exit()

		self.filters = filters

		if kernel_size <= 0:
			print('the kernel size should be a positive number')

		self.kernel_size = kernel_size

		if activation_ is None:
			self.activation = None

		elif activation_ == 'relu':
			self.activation = "relu"

		elif activation_ == 'sigmoid':
			self.activation = "sigmoid"

		elif activation_ == 'tanh':
			self.activation = "tanh"

		else:
			print('This activation function is not supported in this network')
			sys.exit()

		self.activation_ = activation_

		if p_layer is None:
			print("The layer canno't be of type None")
		self.p_layer = p_layer

		self.params_size = (self.filters, self.kernel_size, self.kernel_size, self.p_layer.layero_size[-1])
		self.initial_params = np.random.uniform(low=0.1, high=0.1, size=self.params_size)
		self.trained_params = self.initial_params.copy()
		self.layeri_size = self.p_layer.layero_size
		self.layero_size = (self.p_layer.layero_size[0]-self.kernel_size +1, self.p_layer.layero_size[1] - self.kernel_size +1, filters)
		self.o_layer = None

	def convolution_(self, x, conv_filter):
		result = np.zeros(shape=(x.shape[0], x.shape[1], conv_filter.shape[0]))
		for r in np.uint16(np.arange(self.params_size[1]/2.0, x.shape[0] - self.params_size[1]/2.0 + 1)):
			for c in np.uint16(np.arange(self.params_size[1]/2.0, x.shape[1] - self.params_size[1]/2.0 + 1)):
				if len(x.shape) == 2:
					curr_reg = x[r - np.uint16(np.floor(self.params_size[1]/2.0)):r + np.uint16(np.ceil(self.params_size[1]/2.0)), c - np.uint16(np.floor(self.params_size[1]/2.0)):c + np.uint16(np.ceil(self.params_size[1]/2.0))]
				else:
					curr_reg = x[r - np.uint16(np.floor(self.params_size[1]/2.0)):r + np.uint16(np.ceil(self.params_size[1]/2.0)), c - np.uint16(np.floor(self.params_size[1]/2.0)):c + np.uint16(np.ceil(self.params_size[1]/2.0)), :]
			for filt in range(conv_filter.shape[0]):
				curr_result = curr_reg * conv_filter[filt]
				conv_sum = np.sum(curr_result)
				if self.activation is None:
					result[r, c , filt] = conv_sum
				else:
					result[r, c, filt1] = self.activation(conv_sum)
		result = result[np.uint16(self.params_size[0]/2.0):result.shape[0] - np.uint16(self.params_size[1]/2.0), np.uint16(self.params_size[1]/2.0):result.shape[0] - np.uint16(self.params_size[1]/2.0), :]
		return result

	def convolution(self, x):

		if len(x.shape) != len(self.initial_params.shape) - 1:
			print(len(x.shape))
			print(len(self.initial_params.shape))
			print('The dimension of both input and the filter are not equal !')
			sys.exit()
		elif len(x.shape) > 2 or len(self.initial_params.shape) > 3:
			if x.shape[-1] != self.initial_params.shape[-1]:
				print('The number of channels are not equal !')
				sys.exit()
		elif self.initial_params.shape[1] != self.initial_params[2]:
			print('The filter must be a square matrix !')
			sys.exit()
		elif self.initial_params.shape[1]%2==0:
			print('the size of the filter must be an odd number')
			sys.exit()
		self.o_layer = self.convolution_(x, self.trained_params)



class Pooling():
	def __init__(self, pool_size, p_layer, stride = 2, pool = 'max'):
		self.pool_size = pool_size

		if not(type(pool_size) is int) or pool_size <= 0:
			print('please provide a positive number for the pooling size!')
			sys.exit()
		if stride <= 0:
			print('please provide a positive number for the stride!')
			sys.exit()
		self.stride = stride
		if p_layer is None:
			print('The previous Layer should not be a None Object')
			sys.exit()
		self.p_layer = p_layer
		self.layero_size = self.p_layer.layero_size
		self.layero_size = (np.uint16((self.p_layer.layero_size[0] - self.pool_size +1)/stride + 1), np.uint16((self.p_layer.layero_size[1] - self.pool_size +1)/stride + 1), self.p_layer.layero_size[-1])
		self.o_layer = None
		self.pool_size = pool_size
		self.pool = pool

		def pool(self, x, pool):
			o_pool = np.zeros((np.uint16((x.shape[0] - self.pool_size)/self.stride + 1), np.uint16((x.shape[1] - self.pool_size)/self.stride + 1), x.shape[-1]))
			for map_num in range(x.shape[-1]):
				r1 = 0
				for r in np.arange(0, x.shape[0] - self.pool_size +1, self.stride):
					c1 = 0
					for c in np.arange(0, x.shape[1] - self.pool_size + 1, self.stride):
						if pool == 'max':
							o_pool[r1, c1, map_num] = np.max([x[r:r + self.pool_size, c:c + self.pool_size, map_num]])
						if pool =='average':
							o_pool[r1, c1, map_num] = np.mean([x[r:r + self.pool_size, c:c + self.pool_size, map_num]])
						else:
							print(' Enter <max> or <averge> for the pool option !')
							sys.exit()
						c1 += 1
					r1 += 1
			self.o_layer = o_pool


class Relu:
	def __init__(self, p_layer):
		if p_layer is None:
			print('the previous layer in a None Object !')
			sys.exit()
		self.p_layer = p_layer
		self.layero_size = self.p_layer.layero_size
		self.layeri_size = self.p_layer.layeri_size
		self.o_layer = None

	def relu_layer(self, i_layer):
		self.layero_size = layeri_size
		self.o_layer = relu(i_layer)


class Sigmoid:
	def __init__(self, p_layer):
		if p_layer is None:
			print('The previous Layer should not be a None Object')
			sys.exit()
		self.p_layer = p_layer
		self.layeri_size = self.p_layer.layero_size
		self.layero_size = self.p_layer.layero_size
		self.o_layer = None

	def sigmoid_layer(self, i_layer):
		self.layero_size = i_layer.size
		self.o_layer = sigmoid(i_layer)


class Flatten:
	def __init__(self, p_layer):
		if p_layer is None:
			print('The previous Layer should not be a None Object')
			sys.exit()
		self.p_layer = p_layer
		self.layeri_size = self.p_layer.layero_size
		self.layero_size = functools.reduce(lambda  x , y : x*y, self.p_layer.layero_size)
		self.o_layer = None

	def flatten(self, x):
		self.layero_size = x.size
		self.o_layer = np.ravel(x)


class Dense:
	def __init__(self, neurons, p_layer, activation_):
		self.p_layer = p_layer
		self.neurons = neurons
		if (activation_ == "relu"):
			self.activation = "relu"
		elif (activation_ == "sigmoid"):
			self.activation = "sigmoid"
		elif (activation_ == "softmax"):
			self.activation = "softmax"
		else:
			print('This activation function is not supported !')
			sys.exit()
		self.activation_ = activation_
		if type(self.p_layer.layero_size) in [list, tuple, np.ndarray] and len(self.p_layer.layero_size) > 1:
			print('the input of the dense layer must be an int type !')
			sys.exit()
		self.initial_params = np.random.uniform(low=0.1, high=0.1, size = (self.p_layer.layero_size, self.neurons))
		self.trained_params = self.initial_params.copy()
		self.layeri_size = self.p_layer.layero_size
		self.layero_size = neurons
		self.o_layer = None

	def dense_layer(self, i_layer):
		if self.trained_params is None:
			print('The weights are not trained and they are of a type None !')
			sys.exit()
		s = np.matmul(i_layer, self.trained_params)
		self.o_layer = self.activation(s)
