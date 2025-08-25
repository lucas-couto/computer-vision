import yaml

def load_config():
  conf = yaml.safe_load(open("config.yml"))
  return conf