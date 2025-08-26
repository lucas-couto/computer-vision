from keras import layers, models
from .base_model import BaseModel
from keras.applications import Xception

class XceptionModel(BaseModel):
    def __init__(self, config, data):
        super().__init__(config, data, checkpoint_dir="checkpoints/xception")

    def build_model(self):
        base_model = Xception(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Congelar todas as camadas do modelo base
        base_model.trainable = False
        
        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())

        # Dense 1
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        # Dense 2
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Dense 3
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        # Saída
        model.add(layers.Dense(self.num_classes, activation='sigmoid'))

        return model