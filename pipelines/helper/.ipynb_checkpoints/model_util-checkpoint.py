from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

def define_model(input_shape, num_classes):
        model = Sequential()
        model.add(
        Conv2D(filters=4, kernel_size=(2, 2), strides=1, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=8, kernel_size=(2, 2), strides=1, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model