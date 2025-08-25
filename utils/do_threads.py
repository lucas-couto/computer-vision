import traceback
import gc
import tensorflow as tf
from .load_data import load_data
from .train_and_evaluate import train_and_evaluate

def do_threads(models, config):
    image_data = load_data(config)

    for model_class in models:
        try:
            train_and_evaluate(model_class, config, image_data)
        except Exception as e:
            print(f"Erro no modelo {model_class.__name__}: {e}")
            traceback.print_exc()

    # Limpa image_data da memória
    del image_data
    tf.keras.backend.clear_session()
    gc.collect()
