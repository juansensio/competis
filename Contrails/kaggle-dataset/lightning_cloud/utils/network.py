# from lightning.utilities.network import find_free_network_port
import socket


def find_free_network_port() -> int:
    """Finds a free port on localhost."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.listen(1)
            return sock.getsockname()[1]
    except BaseException:
        return find_free_network_port()
