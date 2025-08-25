import gc
import tensorflow as tf

def train_and_evaluate(model_class, config, data):
    model = model_class(config, data)
    model.train()
    results = model.evaluate()
    print(f"{model_class.__name__} results: {results}")
    
    del model
    tf.keras.backend.clear_session()
    gc.collect()