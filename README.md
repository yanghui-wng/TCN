# TCN
I am studying the code about TCN on github (https://github.com/philipperemy/keras-tcn). The number of parameters of the TCN I calculated is different from the answer of the function "model.summary()". The ansewer of the function "model.summary()" is 153500, but I don't know how to calculate this value.
```python
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
```
