import os
from pathlib import Path


def truthy(val) -> bool:
    if val in [
            "1",
            "true",
            "True",
            True,
    ]:
        return True
    if val in ["0", "false", "False", False, None]:
        return False

    return bool(val)


LIGHTNING_CLOUD_URL = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")

SSL_CA_CERT = os.getenv("REQUESTS_CA_BUNDLE",
                        default=os.getenv("SSL_CERT_FILE", default=None))
VERSION = os.getenv("VERSION", "0.0.1")
DEBUG = truthy(os.getenv("DEBUG"))
CONTEXT = os.getenv("CONTEXT", "staging-3")
LIGHTNING_SETTINGS_PATH = os.getenv(
    'LIGHTNING_SETTINGS_PATH',
    str(Path.home() / '.lightning' / 'settings.json'))
LIGHTNING_CREDENTIAL_PATH = os.getenv(
    'LIGHTNING_CREDENTIAL_PATH',
    str(Path.home() / '.lightning' / 'credentials.json'))

DOT_IGNORE_FILENAME = ".lightningignore"

LEEWAY = 100
IS_DEV_ENV = True


def reset_global_variables() -> None:
    """ Reset the settings from env variables"""
    global DEBUG, CONTEXT, LIGHTNING_CLOUD_URL

    if 'DEBUG' in os.environ:
        DEBUG = truthy(os.environ['DEBUG'])

    if 'GRID_CLUSTER_ID' in os.environ:
        CONTEXT = os.environ['GRID_CLUSTER_ID']

    if 'GRID_URL' in os.environ:
        LIGHTNING_CLOUD_URL = os.environ['GRID_URL']
