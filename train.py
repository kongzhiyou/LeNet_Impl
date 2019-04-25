
import model
from tensorflow.examples.tutorials.mnist import input_data
from keras.optimizers import SGD

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels


depth,height,width = X_train.shape[1],X_train.shape[2],X_train.shape[3]

print(width,height,depth)

model = model.LeNet.build(width,height,depth,10)

opt = SGD(lr=0.01)

model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(X_train,y_train,batch_size=32,epochs=20,validation_data=(X_validation,y_validation),steps_per_epoch=len(X_train)//32,validation_steps=len(X_validation)//32)

model.save('my_model.h5')
