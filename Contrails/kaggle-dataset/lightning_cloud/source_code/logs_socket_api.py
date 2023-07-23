from urllib.parse import urlparse

from lightning_cloud.login import Auth
from lightning_cloud.openapi import ApiClient, AuthServiceApi, V1LoginRequest
import websocket


class LightningLogsSocketAPI:

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client
        self._auth = Auth()
        self._auth.authenticate()
        self._auth_service = AuthServiceApi(api_client)

    def _get_api_token(self):
        token_resp = self._auth_service.auth_service_login(body=V1LoginRequest(
            username=self._auth.username,
            api_key=self._auth.api_key,
        ))
        return token_resp.token

    @staticmethod
    def _socket_url(host: str, project_id: str, app_id: str, token: str,
                    component: str):
        return (f"wss://{host}/v1/projects/{project_id}/apps/{app_id}/logs?"
                f"token={token}&component={component}&follow=true")

    def create_lightning_logs_socket(
        self,
        project_id: str,
        app_id: str,
        component: str,
        on_message_callback,
        on_error_callback=None,
    ) -> websocket.WebSocketApp:
        """
        Creates and returns WebSocketApp to listen to lightning app logs.

        Usage example synchronous:
            def print_log_msg(ws_app, msg):
                print(msg)

            flow_logs_socket = client.create_lightning_logs_socket('project_id', 'app_id', 'flow', print_log_msg)
            flow_socket.run_forever()

        Usage example asynchronous, multiple components:
            def print_log_msg(ws_app, msg):
                print(msg)

            flow_logs_socket = client.create_lightning_logs_socket('project_id', 'app_id', 'flow', print_log_msg)
            work_logs_socket = client.create_lightning_logs_socket('project_id', 'app_id', 'work_1', print_log_msg)

            flow_logs_thread = Thread(target=flow_logs_socket.run_forever)
            work_logs_thread = Thread(target=work_logs_socket.run_forever)

            flow_logs_thread.start()
            work_logs_thread.start()
            .......

            flow_logs_socket.close()
            work_logs_thread.close()

        Parameters
        ----------
        project_id: str
        app_id: str
        component: str
        on_message_callback: function
            Callback object which is called when received data.
            on_message_callback has 2 arguments.
            The 1st argument is the WebSocketApp object.
            The 2nd argument is utf-8 data received from the server.
        on_error_callback: function
            Callback object which is called when we get error.
            on_error has 2 arguments.
            The 1st argument is this class object.
            The 2nd argument is exception object.

        Returns
        -------
        WebSocketApp of the wanted socket
        """
        _token = self._get_api_token()
        clean_ws_host = urlparse(self.api_client.configuration.host).netloc
        socket_url = self._socket_url(
            host=clean_ws_host,
            project_id=project_id,
            app_id=app_id,
            token=_token,
            component=component,
        )

        return websocket.WebSocketApp(socket_url,
                                      on_message=on_message_callback,
                                      on_error=on_error_callback)
