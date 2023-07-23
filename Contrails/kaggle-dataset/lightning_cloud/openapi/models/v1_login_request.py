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

import pprint
import re  # noqa: F401

from typing import TYPE_CHECKING

import six

if TYPE_CHECKING:
    from datetime import datetime
    from lightning_cloud.openapi.models import *


class V1LoginRequest(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {'api_key': 'str', 'duration': 'str', 'username': 'str'}

    attribute_map = {
        'api_key': 'apiKey',
        'duration': 'duration',
        'username': 'username'
    }

    def __init__(self,
                 api_key: 'str' = None,
                 duration: 'str' = None,
                 username: 'str' = None):  # noqa: E501
        """V1LoginRequest - a model defined in Swagger"""  # noqa: E501
        self._api_key = None
        self._duration = None
        self._username = None
        self.discriminator = None
        if api_key is not None:
            self.api_key = api_key
        if duration is not None:
            self.duration = duration
        if username is not None:
            self.username = username

    @property
    def api_key(self) -> 'str':
        """Gets the api_key of this V1LoginRequest.  # noqa: E501


        :return: The api_key of this V1LoginRequest.  # noqa: E501
        :rtype: str
        """
        return self._api_key

    @api_key.setter
    def api_key(self, api_key: 'str'):
        """Sets the api_key of this V1LoginRequest.


        :param api_key: The api_key of this V1LoginRequest.  # noqa: E501
        :type: str
        """

        self._api_key = api_key

    @property
    def duration(self) -> 'str':
        """Gets the duration of this V1LoginRequest.  # noqa: E501


        :return: The duration of this V1LoginRequest.  # noqa: E501
        :rtype: str
        """
        return self._duration

    @duration.setter
    def duration(self, duration: 'str'):
        """Sets the duration of this V1LoginRequest.


        :param duration: The duration of this V1LoginRequest.  # noqa: E501
        :type: str
        """

        self._duration = duration

    @property
    def username(self) -> 'str':
        """Gets the username of this V1LoginRequest.  # noqa: E501


        :return: The username of this V1LoginRequest.  # noqa: E501
        :rtype: str
        """
        return self._username

    @username.setter
    def username(self, username: 'str'):
        """Sets the username of this V1LoginRequest.


        :param username: The username of this V1LoginRequest.  # noqa: E501
        :type: str
        """

        self._username = username

    def to_dict(self) -> dict:
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(lambda x: x.to_dict()
                        if hasattr(x, "to_dict") else x, value))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict") else item,
                        value.items()))
            else:
                result[attr] = value
        if issubclass(V1LoginRequest, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1LoginRequest') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1LoginRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1LoginRequest') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
