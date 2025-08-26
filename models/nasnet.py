from keras import layers, models
from .base_model import BaseModel
from keras.applications import NASNetLarge

class Nasnet(BaseModel):
    def __init__(self, config, data):
        self.best_model_path = "checkpoints/nasnet.keras"
        self.best_weights_path = "checkpoints/nasnet.weights.h5"
        super().__init__(config, data)

    def build_model(self):
        base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False

        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())

        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(self.num_classes, activation='sigmoid'))
        
        return model