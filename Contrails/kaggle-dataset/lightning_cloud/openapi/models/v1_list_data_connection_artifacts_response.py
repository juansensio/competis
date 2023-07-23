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


class V1ListDataConnectionArtifactsResponse(object):
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
        'artifacts': 'list[V1DataConnectionArtifact]',
        'next_page_token': 'str',
        'previous_page_token': 'str'
    }

    attribute_map = {
        'artifacts': 'artifacts',
        'next_page_token': 'nextPageToken',
        'previous_page_token': 'previousPageToken'
    }

    def __init__(self,
                 artifacts: 'list[V1DataConnectionArtifact]' = None,
                 next_page_token: 'str' = None,
                 previous_page_token: 'str' = None):  # noqa: E501
        """V1ListDataConnectionArtifactsResponse - a model defined in Swagger"""  # noqa: E501
        self._artifacts = None
        self._next_page_token = None
        self._previous_page_token = None
        self.discriminator = None
        if artifacts is not None:
            self.artifacts = artifacts
        if next_page_token is not None:
            self.next_page_token = next_page_token
        if previous_page_token is not None:
            self.previous_page_token = previous_page_token

    @property
    def artifacts(self) -> 'list[V1DataConnectionArtifact]':
        """Gets the artifacts of this V1ListDataConnectionArtifactsResponse.  # noqa: E501


        :return: The artifacts of this V1ListDataConnectionArtifactsResponse.  # noqa: E501
        :rtype: list[V1DataConnectionArtifact]
        """
        return self._artifacts

    @artifacts.setter
    def artifacts(self, artifacts: 'list[V1DataConnectionArtifact]'):
        """Sets the artifacts of this V1ListDataConnectionArtifactsResponse.


        :param artifacts: The artifacts of this V1ListDataConnectionArtifactsResponse.  # noqa: E501
        :type: list[V1DataConnectionArtifact]
        """

        self._artifacts = artifacts

    @property
    def next_page_token(self) -> 'str':
        """Gets the next_page_token of this V1ListDataConnectionArtifactsResponse.  # noqa: E501


        :return: The next_page_token of this V1ListDataConnectionArtifactsResponse.  # noqa: E501
        :rtype: str
        """
        return self._next_page_token

    @next_page_token.setter
    def next_page_token(self, next_page_token: 'str'):
        """Sets the next_page_token of this V1ListDataConnectionArtifactsResponse.


        :param next_page_token: The next_page_token of this V1ListDataConnectionArtifactsResponse.  # noqa: E501
        :type: str
        """

        self._next_page_token = next_page_token

    @property
    def previous_page_token(self) -> 'str':
        """Gets the previous_page_token of this V1ListDataConnectionArtifactsResponse.  # noqa: E501


        :return: The previous_page_token of this V1ListDataConnectionArtifactsResponse.  # noqa: E501
        :rtype: str
        """
        return self._previous_page_token

    @previous_page_token.setter
    def previous_page_token(self, previous_page_token: 'str'):
        """Sets the previous_page_token of this V1ListDataConnectionArtifactsResponse.


        :param previous_page_token: The previous_page_token of this V1ListDataConnectionArtifactsResponse.  # noqa: E501
        :type: str
        """

        self._previous_page_token = previous_page_token

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
        if issubclass(V1ListDataConnectionArtifactsResponse, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1ListDataConnectionArtifactsResponse') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1ListDataConnectionArtifactsResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1ListDataConnectionArtifactsResponse') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
