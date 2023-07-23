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


class V1NetworkConfig(object):
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
    swagger_types = {
        'enable': 'bool',
        'host': 'str',
        'name': 'str',
        'port': 'int'
    }

    attribute_map = {
        'enable': 'enable',
        'host': 'host',
        'name': 'name',
        'port': 'port'
    }

    def __init__(self,
                 enable: 'bool' = None,
                 host: 'str' = None,
                 name: 'str' = None,
                 port: 'int' = None):  # noqa: E501
        """V1NetworkConfig - a model defined in Swagger"""  # noqa: E501
        self._enable = None
        self._host = None
        self._name = None
        self._port = None
        self.discriminator = None
        if enable is not None:
            self.enable = enable
        if host is not None:
            self.host = host
        if name is not None:
            self.name = name
        if port is not None:
            self.port = port

    @property
    def enable(self) -> 'bool':
        """Gets the enable of this V1NetworkConfig.  # noqa: E501


        :return: The enable of this V1NetworkConfig.  # noqa: E501
        :rtype: bool
        """
        return self._enable

    @enable.setter
    def enable(self, enable: 'bool'):
        """Sets the enable of this V1NetworkConfig.


        :param enable: The enable of this V1NetworkConfig.  # noqa: E501
        :type: bool
        """

        self._enable = enable

    @property
    def host(self) -> 'str':
        """Gets the host of this V1NetworkConfig.  # noqa: E501


        :return: The host of this V1NetworkConfig.  # noqa: E501
        :rtype: str
        """
        return self._host

    @host.setter
    def host(self, host: 'str'):
        """Sets the host of this V1NetworkConfig.


        :param host: The host of this V1NetworkConfig.  # noqa: E501
        :type: str
        """

        self._host = host

    @property
    def name(self) -> 'str':
        """Gets the name of this V1NetworkConfig.  # noqa: E501


        :return: The name of this V1NetworkConfig.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this V1NetworkConfig.


        :param name: The name of this V1NetworkConfig.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def port(self) -> 'int':
        """Gets the port of this V1NetworkConfig.  # noqa: E501


        :return: The port of this V1NetworkConfig.  # noqa: E501
        :rtype: int
        """
        return self._port

    @port.setter
    def port(self, port: 'int'):
        """Sets the port of this V1NetworkConfig.


        :param port: The port of this V1NetworkConfig.  # noqa: E501
        :type: int
        """

        self._port = port

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
        if issubclass(V1NetworkConfig, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1NetworkConfig') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1NetworkConfig):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1NetworkConfig') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
