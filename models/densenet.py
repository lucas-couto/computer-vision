from keras import layers, models
from .base_model import BaseModel
from keras.applications import DenseNet121

class Densenet(BaseModel):
    def __init__(self, config, data):
        self.best_model_path = "checkpoints/densenet.keras"
        self.best_weights_path = "checkpoints/densenet.weights.h5"
        super().__init__(config, data)

    def build_model(self):
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False  # Pode destravar depois de algumas épocas para fine-tuning
        
        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        
        # Camada 1
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        # Camada 2
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        # Camada 3
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        # Saída
        model.add(layers.Dense(self.num_classes, activation='sigmoid'))

        return model