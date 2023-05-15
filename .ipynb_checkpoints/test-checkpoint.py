from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

input_shape = (448, 448, 3)
inputs = Input(input_shape)



x = Conv2D(filters = 16,
           kernel_size = (3, 3), 
           padding='same', 
           name='convolutional_0', 
           use_bias=False,
           kernel_regularizer=l2(5e-4), 
           trainable=False
)(inputs)


model = Model(inputs=inputs, outputs=x)

print(model.summary())

tf.keras.utils.plot_model(model,
                                  to_file='test.png',
                                  show_shapes=True,
                                  show_layer_names=True)