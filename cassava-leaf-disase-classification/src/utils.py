import yaml
import uuid
import io

def parse_config_file(config_file):
    # parse yaml file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    # generate unique id and save file
    uid = uuid.uuid4().hex
    name = config_file.split('.')[0]
    new_name = f'{name}-{uid}.yml'
    with io.open(new_name, 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)
    # add file name
    config['name'] = uid
    return config