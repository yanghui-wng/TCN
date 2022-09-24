# TCN
I am studying the code about TCN on github (https://github.com/philipperemy/keras-tcn). The number of parameters of the TCN I calculated is different from the answer of the function "model.summary()". The parameter of TCN layer that calculated by the function "model.summary()" is 153500, but I don't know how to calculate this value and I am trying to calculate the value, but the result is 153000.
## Code
```python
# design network
batch_size = None
model = Sequential()
input_layer = Input(batch_shape=(batch_size,1,7))
model.add(input_layer)
model.add(TCN(nb_filters=100, #Integer. The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
        kernel_size=3, #Integer. The size of the kernel to use in each convolutional layer.
        nb_stacks=1,   #The number of stacks of residual blocks to use.
        dilations=(1,2,4), #List/Tuple. A dilation list. Example is: [1, 2, 4, 8, 16, 32, 64].
        padding='causal',
        use_skip_connections=False, 
        dropout_rate=0.1,
        return_sequences=False,
        activation='relu', 
        kernel_initializer='he_normal', 
        use_batch_norm=False, 
        use_layer_norm=False, 
        ))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(1))
model.add(LeakyReLU(alpha=0.3))
model.compile(loss='mse', optimizer='adam')
model.summary()
```
## Output
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
tcn_2 (TCN)                  (None, 100)               153500    
_________________________________________________________________
dense_8 (Dense)              (None, 64)                6464      
_________________________________________________________________
leaky_re_lu_8 (LeakyReLU)    (None, 64)                0         
_________________________________________________________________
dense_9 (Dense)              (None, 32)                2080      
_________________________________________________________________
leaky_re_lu_9 (LeakyReLU)    (None, 32)                0         
_________________________________________________________________
dense_10 (Dense)             (None, 16)                528       
_________________________________________________________________
leaky_re_lu_10 (LeakyReLU)   (None, 16)                0         
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 17        
_________________________________________________________________
leaky_re_lu_11 (LeakyReLU)   (None, 1)                 0         
=================================================================
Total params: 162,589
Trainable params: 162,589
Non-trainable params: 0
_________________________________________________________________
```
