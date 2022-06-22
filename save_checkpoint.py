import tensorflow as tf
import os
from tensorflow import keras

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './临时文件/mnist.ckpt'
# 生成ckpt的同时会生成索引文件
if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------load the model----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                              save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

