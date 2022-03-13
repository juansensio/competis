import collections


def get_patch(sample, path='data'):
    country_id = str(sample)[0]
    country = 'fr' if country_id == '1' else 'us'
    subfolder = str(sample)[-2:]
    subsubfolder = str(sample)[-4:-2]
    return path + '/patches-' + country + '/' + subfolder + '/' + subsubfolder


def get_patch_image(sample, image, path='data'):
    patch = get_patch(sample, path)
    return patch + '/' + str(sample) + '_' + image


def get_patch_rgb(sample, path='data'):
    return get_patch_image(sample, 'rgb.jpg', path)


def get_patch_nir(sample, path='data'):
    return get_patch_image(sample, 'near_ir.jpg', path)


def get_patch_altitude(sample, path='data'):
    return get_patch_image(sample, 'altitude.tif', path)


def get_patch_landcover(sample, path='data'):
    return get_patch_image(sample, 'landcover.tif', path)


def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source
