import click

from lightning_cloud.login import Auth


@click.group()
def main():
    pass


@main.command()
def login():
    """Authorize the CLI to access Grid AI resources for a particular user.
    Use login command to force authenticate,
    a web browser will open to complete the authentication.
    """
    auth = Auth()
    auth.clear()
    auth._run_server()


@main.command()
def logout():
    """Logout from LightningCloud"""
    Auth.clear()
