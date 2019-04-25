
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

class LeNet:
    def build(width,height,depth,classes,weightsPath=None):

        model = Sequential()


        #Conv1
        model.add(Conv2D(filters=20,
                         kernel_size=(5,5),
                         strides=(1,1),
                         padding='same',
                         input_shape=(height,depth,width)))
        #Activation
        model.add(Activation(activation='relu'))
        print('act_1')
        #MaxPooling
        model.add(MaxPool2D(pool_size=(2,2),strides=2,padding='same'))

        #Conv2
        model.add(Conv2D(50,(5,5),padding='same'))
        #Activation
        model.add(Activation('relu'))
        print('act_2')
        #Maxpooling
        model.add(MaxPool2D((2,2),2,padding='same'))

        #Conv3
        model.add(Flatten()) #将特征图展平(1,1)filters,stride(1,1)

        #Dense
        model.add(Dense(500))
        model.add(Activation('relu'))
        print('act_3')

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        print('act_4')
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model


