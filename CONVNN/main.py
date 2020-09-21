import numpy as np 
from layer_types import Inp, Conv, Pooling, Dense, Relu, Flatten
from model import Model

print('The Program starts Now ...')

print('Uploading Data ...')

data_x = np.load('datasets/dataset_inputs.npy')
data_y = np.load('datasets/dataset_outputs.npy')
train_x = data_x[:60, :, :, :]
train_y = data_y[:60]
test_x = data_x[60:, :, :, :]

print('Create the Architecture ...')

s_shape = train_x.shape[1:]
classes = 4
i_layer = Inp(s_shape)
conv_l1 = Conv(2, 3, i_layer, None)
activ_l1 = Sigmoid(conv_l1)
avr_pool = Pooling(2, activ_l1, stride = 2, pool = 'average')

conv_l2 = Conv(3, 3, avr_pool, None)
activ_l2 = Relu(conv_l2)
max_pool = Pooling(2, activ_l2, stride = 2, pool = 'max')

conv_l3 = Conv(1, 3, max_pool, None)
activ_l3 = Relu(conv_l3)
avr_pool2 = Pooling(2, activ_l3, stride = 2, pool = 'average')

inp = Flatten(avr_pool2)

dense_l1 = Dense(150, inp,'relu')
dense_l2 = Dense(300, dense_l1, 'relu')
dense_l3 = Dense(classes, dense_l2,'relu')

model = Model(dense_l3, epochs = 1000, learning_rate = 0.02)

print('Model Summary ...')

model.summary()

print('The training starts Now ...')

model.train(train_x, train_y)

print('The Predictions starts Now ...')

preds = model.predict(test_x)

print(preds)

