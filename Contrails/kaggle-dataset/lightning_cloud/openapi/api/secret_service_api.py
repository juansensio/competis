# coding: utf-8
"""
    external/v1/auth_service.proto

    No description provided (generated by Swagger Codegen https://github.com/swagger-api/swagger-codegen)  # noqa: E501

    OpenAPI spec version: version not set

    Generated by: https://github.com/swagger-api/swagger-codegen.git

    NOTE
    ----
    standard swagger-codegen-cli for this python client has been modified
    by custom templates. The purpose of these templates is to include
    typing information in the API and Model code. Please refer to the
    main grid repository for more info
"""

from __future__ import absolute_import

import re  # noqa: F401
from typing import TYPE_CHECKING, Any

# python 2 and python 3 compatibility library
import six

from lightning_cloud.openapi.api_client import ApiClient

if TYPE_CHECKING:
    from datetime import datetime
    from lightning_cloud.openapi.models import *


class SecretServiceApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def secret_service_create_secret(self, body: 'ProjectIdSecretsBody',
                                     project_id: 'str',
                                     **kwargs) -> 'V1Secret':  # noqa: E501
        """secret_service_create_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_create_secret(body, project_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param ProjectIdSecretsBody body: (required)
        :param str project_id: (required)
        :return: V1Secret
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.secret_service_create_secret_with_http_info(
                body, project_id, **kwargs)  # noqa: E501
        else:
            (data) = self.secret_service_create_secret_with_http_info(
                body, project_id, **kwargs)  # noqa: E501
            return data

    def secret_service_create_secret_with_http_info(
            self, body: 'ProjectIdSecretsBody', project_id: 'str',
            **kwargs) -> 'V1Secret':  # noqa: E501
        """secret_service_create_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_create_secret_with_http_info(body, project_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param ProjectIdSecretsBody body: (required)
        :param str project_id: (required)
        :return: V1Secret
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body', 'project_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method secret_service_create_secret" %
                                key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or params['body'] is None):
            raise ValueError(
                "Missing the required parameter `body` when calling `secret_service_create_secret`"
            )  # noqa: E501
        # verify the required parameter 'project_id' is set
        if ('project_id' not in params or params['project_id'] is None):
            raise ValueError(
                "Missing the required parameter `project_id` when calling `secret_service_create_secret`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'project_id' in params:
            path_params['projectId'] = params['project_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'body' in params:
            body_params = params['body']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params[
            'Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
                ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            '/v1/projects/{projectId}/secrets',
            'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1Secret',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def secret_service_delete_secret(
            self, project_id: 'str', id: 'str',
            **kwargs) -> 'V1DeleteSecretResponse':  # noqa: E501
        """secret_service_delete_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_delete_secret(project_id, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str project_id: (required)
        :param str id: (required)
        :return: V1DeleteSecretResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.secret_service_delete_secret_with_http_info(
                project_id, id, **kwargs)  # noqa: E501
        else:
            (data) = self.secret_service_delete_secret_with_http_info(
                project_id, id, **kwargs)  # noqa: E501
            return data

    def secret_service_delete_secret_with_http_info(
            self, project_id: 'str', id: 'str',
            **kwargs) -> 'V1DeleteSecretResponse':  # noqa: E501
        """secret_service_delete_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_delete_secret_with_http_info(project_id, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str project_id: (required)
        :param str id: (required)
        :return: V1DeleteSecretResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['project_id', 'id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method secret_service_delete_secret" %
                                key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'project_id' is set
        if ('project_id' not in params or params['project_id'] is None):
            raise ValueError(
                "Missing the required parameter `project_id` when calling `secret_service_delete_secret`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if ('id' not in params or params['id'] is None):
            raise ValueError(
                "Missing the required parameter `id` when calling `secret_service_delete_secret`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'project_id' in params:
            path_params['projectId'] = params['project_id']  # noqa: E501
        if 'id' in params:
            path_params['id'] = params['id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            '/v1/projects/{projectId}/secrets/{id}',
            'DELETE',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1DeleteSecretResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def secret_service_get_secret(self, project_id: 'str', id: 'str',
                                  **kwargs) -> 'V1Secret':  # noqa: E501
        """secret_service_get_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_get_secret(project_id, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str project_id: (required)
        :param str id: (required)
        :return: V1Secret
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.secret_service_get_secret_with_http_info(
                project_id, id, **kwargs)  # noqa: E501
        else:
            (data) = self.secret_service_get_secret_with_http_info(
                project_id, id, **kwargs)  # noqa: E501
            return data

    def secret_service_get_secret_with_http_info(
            self, project_id: 'str', id: 'str',
            **kwargs) -> 'V1Secret':  # noqa: E501
        """secret_service_get_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_get_secret_with_http_info(project_id, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str project_id: (required)
        :param str id: (required)
        :return: V1Secret
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['project_id', 'id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method secret_service_get_secret" % key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'project_id' is set
        if ('project_id' not in params or params['project_id'] is None):
            raise ValueError(
                "Missing the required parameter `project_id` when calling `secret_service_get_secret`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if ('id' not in params or params['id'] is None):
            raise ValueError(
                "Missing the required parameter `id` when calling `secret_service_get_secret`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'project_id' in params:
            path_params['projectId'] = params['project_id']  # noqa: E501
        if 'id' in params:
            path_params['id'] = params['id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            '/v1/projects/{projectId}/secrets/{id}',
            'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1Secret',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def secret_service_list_secrets(
            self, project_id: 'str',
            **kwargs) -> 'V1ListSecretsResponse':  # noqa: E501
        """secret_service_list_secrets  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_list_secrets(project_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str project_id: (required)
        :return: V1ListSecretsResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.secret_service_list_secrets_with_http_info(
                project_id, **kwargs)  # noqa: E501
        else:
            (data) = self.secret_service_list_secrets_with_http_info(
                project_id, **kwargs)  # noqa: E501
            return data

    def secret_service_list_secrets_with_http_info(
            self, project_id: 'str',
            **kwargs) -> 'V1ListSecretsResponse':  # noqa: E501
        """secret_service_list_secrets  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_list_secrets_with_http_info(project_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str project_id: (required)
        :return: V1ListSecretsResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['project_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method secret_service_list_secrets" % key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'project_id' is set
        if ('project_id' not in params or params['project_id'] is None):
            raise ValueError(
                "Missing the required parameter `project_id` when calling `secret_service_list_secrets`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'project_id' in params:
            path_params['projectId'] = params['project_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            '/v1/projects/{projectId}/secrets',
            'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1ListSecretsResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def secret_service_update_secret(self, body: 'SecretsIdBody',
                                     project_id: 'str', id: 'str',
                                     **kwargs) -> 'V1Secret':  # noqa: E501
        """secret_service_update_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_update_secret(body, project_id, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param SecretsIdBody body: (required)
        :param str project_id: (required)
        :param str id: (required)
        :return: V1Secret
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.secret_service_update_secret_with_http_info(
                body, project_id, id, **kwargs)  # noqa: E501
        else:
            (data) = self.secret_service_update_secret_with_http_info(
                body, project_id, id, **kwargs)  # noqa: E501
            return data

    def secret_service_update_secret_with_http_info(
            self, body: 'SecretsIdBody', project_id: 'str', id: 'str',
            **kwargs) -> 'V1Secret':  # noqa: E501
        """secret_service_update_secret  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.secret_service_update_secret_with_http_info(body, project_id, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param SecretsIdBody body: (required)
        :param str project_id: (required)
        :param str id: (required)
        :return: V1Secret
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body', 'project_id', 'id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method secret_service_update_secret" %
                                key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or params['body'] is None):
            raise ValueError(
                "Missing the required parameter `body` when calling `secret_service_update_secret`"
            )  # noqa: E501
        # verify the required parameter 'project_id' is set
        if ('project_id' not in params or params['project_id'] is None):
            raise ValueError(
                "Missing the required parameter `project_id` when calling `secret_service_update_secret`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if ('id' not in params or params['id'] is None):
            raise ValueError(
                "Missing the required parameter `id` when calling `secret_service_update_secret`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'project_id' in params:
            path_params['projectId'] = params['project_id']  # noqa: E501
        if 'id' in params:
            path_params['id'] = params['id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'body' in params:
            body_params = params['body']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params[
            'Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
                ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        return self.api_client.call_api(
            '/v1/projects/{projectId}/secrets/{id}',
            'PUT',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1Secret',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)