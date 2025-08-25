from keras import layers, models
from keras.optimizers import Adam
from utils.save_results import save_results
from tensorflow.keras.models import load_model
from keras.metrics import Precision, Recall, AUC
from utils.find_best_threshold import find_best_threshold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report

class Cnn():
    def __init__(self, config, data):
        self.epochs = config['training']['epochs']
        self.train_dir = config['paths']['train_dir']
        self.valid_dir = config['paths']['valid_dir']
        self.num_classes = config['model']['num_classes']
        self.batch_size = config['training']['batch_size']
        self.input_shape = tuple(config['model']['input_shape'])
        self.learning_rate = config['training']['learning_rate']
        self.patience_early_stop = config['patient']['early_stop']
        self.patience_reduce_lr_plateau = config['patient']['reduce_lr_plateau']
        self.train_data, self.train_labels, self.validation_data, self.validation_labels = data
        self.monitor_metric = "val_loss" if self.validation_data is not None else "loss"

        self.best_model_path = "checkpoints/cnn.keras"
        self.model = self.build_model()


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

        # Pooling global + Flatten
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Flatten())

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

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )

        return model

    def train(self):
        model_checkpoint = ModelCheckpoint(
            filepath=self.best_model_path,
            monitor=self.monitor_metric,
            save_best_only=True,
            mode="min" if "loss" in self.monitor_metric or "error" in self.monitor_metric else "max",
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor=self.monitor_metric,
            patience=self.patience_early_stop,
            mode="min" if "loss" in self.monitor_metric or "error" in self.monitor_metric else "max",
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr_plateau = ReduceLROnPlateau(
            monitor=self.monitor_metric,
            factor=0.5,
            patience=self.patience_reduce_lr_plateau,
            min_lr=1e-6,
            mode="min" if "loss" in self.monitor_metric or "error" in self.monitor_metric else "max",
            verbose=1
        )
        
        self.model.fit(
            self.train_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=self.validation_data,
            callbacks=[model_checkpoint, early_stopping, reduce_lr_plateau]
        )

    def evaluate(self):
        # 1) Carrega o melhor modelo salvo pelo checkpoint
        best_model = load_model(self.best_model_path)

        # 2) Probabilidades no conjunto de validação
        proba = best_model.predict(self.validation_data, verbose=0)
        if proba.ndim == 2 and proba.shape[1] == 1:
            proba = proba.ravel()

        # 3) Encontra o melhor limiar (para problema binário)
        best_threshold = find_best_threshold(best_model, self.validation_data, self.validation_labels)

        # 4) Converte para rótulos com o limiar ótimo
        preds = (proba > best_threshold).astype("int32")

        # 5) Avalia métricas Keras de uma vez
        eval_vals = best_model.evaluate(self.validation_data, verbose=0)
        # Por padrão: [loss, accuracy, precision, recall, auc] (na mesma ordem do compile)
        loss = float(eval_vals[0])
        accuracy = float(eval_vals[1])

        # 6) Métricas do sklearn
        precision = precision_score(self.validation_labels, preds, average='binary')
        recall = recall_score(self.validation_labels, preds, average='binary')
        auc = roc_auc_score(self.validation_labels, proba)  # AUC com probabilidades

        print(classification_report(self.validation_labels, preds, target_names=['fake', 'real']))

        data = {
            'Model': 'cnn',
            'Loss': loss,
            'Accuracy': accuracy,
            'Precision': float(precision),
            'Recall': float(recall),
            'AUC': float(auc),
            'Threshold': float(best_threshold),
            'Checkpoint': self.best_model_path
        }
        save_results("cnn", data)
        return data
    
    def predict(self, X):
        return self.model.predict(X)
