import os
import tensorflow as tf
from tensorflow import keras
import keras.api._v2.keras as KerasAPI
import tensorflow_hub
import config


class Model:
    def __init__(self):
        weights_path = os.path.join(
            os.path.dirname(__file__), r"model/weights.hdf5")
        self.callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="auto",
        )
        self.model = self.create_model()
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print("Weights Loaded")

    def create_model(self):
        model = KerasAPI.Sequential()
        model.add(KerasAPI.layers.Embedding(
            config.vocab_size, 64, input_length=config.pad_length, name="embeddings"))
        model.add(KerasAPI.layers.Flatten())
        for units in [128, 64, 32]:
            model.add(KerasAPI.layers.Dense(
                units, activation=KerasAPI.activations.relu))
            model.add(KerasAPI.layers.Dropout(0.2))
        model.add(KerasAPI.layers.Dense(
            1, activation=KerasAPI.activations.sigmoid))
        model.compile(optimizer=KerasAPI.optimizers.Adam(),
                      loss=KerasAPI.losses.binary_crossentropy,
                      metrics=["accuracy"])
        return model

    def fit(self, x, y, x_test, y_test, epochs=1):
        self.model.fit(x, y, epochs=epochs,
                       validation_data=(x_test, y_test), callbacks=[self.callback])
        return

    def predict(self, x):
        return self.model.predict(x, verbose=0)
