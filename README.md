# Binary_Classification
Binary classification 'sigmoid' activation layer is good for output layer. For multiclass, use 'softmax' for output layer.

When dealing with images we add Convulutional neural networks i.e. Con2d layer. For stable result we can add a 'batch normalization' layer for every convolutional network.

For binary classification, we compile using 'binary_crossentropy' loss, optimizer is 'rmsprop' abd 'Adam'. For multi class classification , we use 'categorical_crosstropy' loss.

In compile, model.compile, the metrics we are using 'accuracy'.
