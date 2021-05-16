import yaml

def parse_config(config_file):
    # parse yaml file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config