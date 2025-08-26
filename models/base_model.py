# base_model.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from keras.optimizers import Adam
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report

from utils.save_results import save_results
from utils.find_best_threshold import find_best_threshold


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any], data: Tuple[Any, Any, Any, Any]):
        # Config
        self.epochs = config['training']['epochs']
        self.num_classes = config['model']['num_classes']
        self.batch_size = config['training']['batch_size']
        self.input_shape = tuple(config['model']['input_shape'])
        self.learning_rate = config['training']['learning_rate']
        self.patience_early_stop = config['patient']['early_stop']
        self.patience_reduce_lr_plateau = config['patient']['reduce_lr_plateau']

        (self.train_data,
         self.train_labels,
         self.validation_data,
         self.validation_labels) = data

        self.monitor_metric = "val_loss" if self.validation_data is not None else "loss"

        self.best_model_path = "checkpoints/model.keras"
        self.best_weights_path = "checkpoints/model.weights.h5"

        self.model = self.build_model()
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )

    @abstractmethod
    def build_model(self):
        """Retorne um keras.Model já com input_shape aplicado nas primeiras camadas."""
        ...

    def _make_callbacks(self):
        mode = "min" if "loss" in self.monitor_metric or "error" in self.monitor_metric else "max"

        ckpt = ModelCheckpoint(
            filepath=self.best_weights_path,
            monitor=self.monitor_metric,
            save_best_only=True,
            save_weights_only=True,
            mode=mode,
            verbose=1
        )

        es = EarlyStopping(
            monitor=self.monitor_metric,
            patience=self.patience_early_stop,
            mode=mode,
            restore_best_weights=True,
            verbose=1
        )

        rlrop = ReduceLROnPlateau(
            monitor=self.monitor_metric,
            factor=0.5,
            patience=self.patience_reduce_lr_plateau,
            min_lr=1e-6,
            mode=mode,
            verbose=1
        )

        return [ckpt, es, rlrop]

    def train(self):
        self.model.fit(
            self.train_data,
            validation_data=self.validation_data,
            validation_freq=2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self._make_callbacks(),
            verbose=1
        )

    def evaluate(self):
        # garante pesos ótimos
        self.model.load_weights(self.best_weights_path)

        # probabilidades no conjunto de validação
        proba = self.model.predict(self.validation_data, verbose=0).ravel()

        # limiar ótimo (binário)
        best_threshold = find_best_threshold(self.model, self.validation_data, self.validation_labels)

        # rótulos com limiar ótimo
        preds = (proba > best_threshold).astype("int32")

        # métricas Keras
        eval_vals = self.model.evaluate(self.validation_data, verbose=0)
        loss = float(eval_vals[0])
        accuracy = float(eval_vals[1])

        # métricas sklearn
        precision = precision_score(self.validation_labels, preds, average='binary', zero_division=0)
        recall = recall_score(self.validation_labels, preds, average='binary', zero_division=0)
        auc = roc_auc_score(self.validation_labels, proba)

        print(classification_report(self.validation_labels, preds, target_names=['fake', 'real']))

        data = {
            'Model': self.__class__.__name__.lower(),
            'Loss': loss,
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall': float(recall),
            'AUC': float(auc),
            'Threshold': float(best_threshold),
            'Checkpoint': self.best_model_path
        }
        save_results(self.__class__.__name__.lower(), data)
        return data

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        self.model.load_weights(self.best_weights_path)
        self.model.save(self.best_model_path)
        print(f"Modelo completo salvo em: {self.best_model_path}")
