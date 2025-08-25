from .load_config import load_config
from .preprocess_images import preprocess_images
from .do_threads import do_threads
from .get_models import get_models

def run_all_models():
  config = load_config()
  models = get_models()
  preprocess_images(config)
  do_threads(models, config)