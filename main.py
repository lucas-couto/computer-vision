import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from utils.run_all_models import run_all_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.optimizer.set_jit(False)
mixed_precision.set_global_policy('mixed_float16')
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    
def main():
    run_all_models()

if __name__ == "__main__":
    main()
