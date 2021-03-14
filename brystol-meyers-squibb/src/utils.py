from pathlib import Path
import yaml
import uuid
import io
import Levenshtein

def get_image_path(image_id, path=Path('data'), mode="train"):
    return path / mode / image_id[0] / image_id[1] / image_id[2] / f'{image_id}.png'
    #return path / 'bms100' / mode / f'{image_id}.png'

def parse_config_file(config_file, save_new=False):
    # parse yaml file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    # generate unique id and save file
    if save_new:
        uid = uuid.uuid4().hex
        name = config_file.split('.')[0]
        new_name = f'{name}-{uid}.yml'
        with io.open(new_name, 'w', encoding='utf8') as outfile:
            yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)
        # add file name
        config['name'] = uid
    return config

def Levenshtein_dist(y_hat, y):
    ld = []
    y_pred = torch.argmax(pred, axis=2)
    for pred, gt in zip(y_pred, y):
        # convert pred to string
        # convert gt to string
        ls.append(Levenshtein.distance(gt, pred))
    return np.mean(ld)