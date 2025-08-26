# models/cnn.py
from keras import layers, models
from .base_model import BaseModel

class Cnn(BaseModel):
    def __init__(self, config, data):
        super().__init__(config, data, checkpoint_dir="checkpoints/cnn")

    def build_model(self):
        model = models.Sequential()
        # Bloco 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        # Bloco 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        # Bloco 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        # Bloco 4
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        # Bloco 5
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), padding='same'))

        # Pooling + Dense
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Flatten())

        model.add(layers.Dense(1024, activation='relu')); model.add(layers.BatchNormalization()); model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512,  activation='relu')); model.add(layers.BatchNormalization()); model.add(layers.Dropout(0.4))
        model.add(layers.Dense(256,  activation='relu')); model.add(layers.BatchNormalization()); model.add(layers.Dropout(0.3))

        # Saída binária
        model.add(layers.Dense(self.num_classes, activation='sigmoid'))
        return model
