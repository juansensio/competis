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


class AuthServiceApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def auth_service_get_user(self,
                              **kwargs) -> 'V1GetUserResponse':  # noqa: E501
        """auth_service_get_user  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_get_user(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: V1GetUserResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.auth_service_get_user_with_http_info(
                **kwargs)  # noqa: E501
        else:
            (data) = self.auth_service_get_user_with_http_info(
                **kwargs)  # noqa: E501
            return data

    def auth_service_get_user_with_http_info(
            self, **kwargs) -> 'V1GetUserResponse':  # noqa: E501
        """auth_service_get_user  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_get_user_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: V1GetUserResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = []  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method auth_service_get_user" % key)
            params[key] = val
        del params['kwargs']

        collection_formats = {}

        path_params = {}

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
            '/v1/auth/user',
            'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1GetUserResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def auth_service_login(self, body: 'V1LoginRequest',
                           **kwargs) -> 'V1LoginResponse':  # noqa: E501
        """auth_service_login  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_login(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1LoginRequest body: (required)
        :return: V1LoginResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.auth_service_login_with_http_info(
                body, **kwargs)  # noqa: E501
        else:
            (data) = self.auth_service_login_with_http_info(
                body, **kwargs)  # noqa: E501
            return data

    def auth_service_login_with_http_info(
            self, body: 'V1LoginRequest',
            **kwargs) -> 'V1LoginResponse':  # noqa: E501
        """auth_service_login  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_login_with_http_info(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1LoginRequest body: (required)
        :return: V1LoginResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method auth_service_login" % key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or params['body'] is None):
            raise ValueError(
                "Missing the required parameter `body` when calling `auth_service_login`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}

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
            '/v1/auth/login',
            'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1LoginResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def auth_service_logout(self, body: 'V1LogoutRequest',
                            **kwargs) -> 'V1LogoutResponse':  # noqa: E501
        """auth_service_logout  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_logout(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1LogoutRequest body: (required)
        :return: V1LogoutResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.auth_service_logout_with_http_info(
                body, **kwargs)  # noqa: E501
        else:
            (data) = self.auth_service_logout_with_http_info(
                body, **kwargs)  # noqa: E501
            return data

    def auth_service_logout_with_http_info(
            self, body: 'V1LogoutRequest',
            **kwargs) -> 'V1LogoutResponse':  # noqa: E501
        """auth_service_logout  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_logout_with_http_info(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1LogoutRequest body: (required)
        :return: V1LogoutResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method auth_service_logout" % key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or params['body'] is None):
            raise ValueError(
                "Missing the required parameter `body` when calling `auth_service_logout`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}

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
            '/v1/auth/logout',
            'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1LogoutResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def auth_service_refresh(self, body: 'V1RefreshRequest',
                             **kwargs) -> 'V1RefreshResponse':  # noqa: E501
        """auth_service_refresh  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_refresh(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1RefreshRequest body: (required)
        :return: V1RefreshResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.auth_service_refresh_with_http_info(
                body, **kwargs)  # noqa: E501
        else:
            (data) = self.auth_service_refresh_with_http_info(
                body, **kwargs)  # noqa: E501
            return data

    def auth_service_refresh_with_http_info(
            self, body: 'V1RefreshRequest',
            **kwargs) -> 'V1RefreshResponse':  # noqa: E501
        """auth_service_refresh  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_refresh_with_http_info(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1RefreshRequest body: (required)
        :return: V1RefreshResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method auth_service_refresh" % key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or params['body'] is None):
            raise ValueError(
                "Missing the required parameter `body` when calling `auth_service_refresh`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}

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
            '/v1/auth/refresh',
            'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1RefreshResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def auth_service_update_user(
            self, body: 'V1UpdateUserRequest',
            **kwargs) -> 'V1GetUserResponse':  # noqa: E501
        """TODO: change `GetUserResponse` to `User`  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_update_user(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1UpdateUserRequest body: (required)
        :return: V1GetUserResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.auth_service_update_user_with_http_info(
                body, **kwargs)  # noqa: E501
        else:
            (data) = self.auth_service_update_user_with_http_info(
                body, **kwargs)  # noqa: E501
            return data

    def auth_service_update_user_with_http_info(
            self, body: 'V1UpdateUserRequest',
            **kwargs) -> 'V1GetUserResponse':  # noqa: E501
        """TODO: change `GetUserResponse` to `User`  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.auth_service_update_user_with_http_info(body, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param V1UpdateUserRequest body: (required)
        :return: V1GetUserResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError("Got an unexpected keyword argument '%s'"
                                " to method auth_service_update_user" % key)
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or params['body'] is None):
            raise ValueError(
                "Missing the required parameter `body` when calling `auth_service_update_user`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}

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
            '/v1/auth/user',
            'PUT',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='V1GetUserResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)