import numpy as np


def layer_params(model, initialize = True):
	net_params = []
	layer = model.l_layer
	while 'p_layer' in layer.__init__.__code__.co_varnames:
		if type(layer) in [Conv, Dense]:
			if initialize == True:
				net_params.append(layer.initial_params)
			if initialize == False:
				net_params.append(layer.trained_params)
		layer = layer.p_layer
	if not(type(layer) is Inp):
		print('The first layer must be an Input Layer')
		sys.exit()
	net_params.reverse()
	return np.array(net_params)


def layer_params_matrix(model, params):
	net_params = []
	srt = 0
	layer = model.l_layer
	params = params[: : -1]
	while 'p_layer' in layer.__init__.__code__.co_varnames:
		if type(layer) in [Conv, Dense]:
			layer_params_shape = layer.initial_params.shape
			layer_params_size = layer.initial_params.size
			params = params[str:str + layer_params_size]
			matrix = np.reshape(params, newshape=(layer_params_shape))
			net_params.append(matrix)
			str = str + layer_params_size
		layer = layer.p_layer
	net_params.reverse()
	return np.array(net_params)
