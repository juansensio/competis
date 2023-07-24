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


class V1ProjectSettings(object):
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
        'preferred_cluster': 'str',
        'same_compute_on_resume': 'bool'
    }

    attribute_map = {
        'preferred_cluster': 'preferredCluster',
        'same_compute_on_resume': 'sameComputeOnResume'
    }

    def __init__(self,
                 preferred_cluster: 'str' = None,
                 same_compute_on_resume: 'bool' = None):  # noqa: E501
        """V1ProjectSettings - a model defined in Swagger"""  # noqa: E501
        self._preferred_cluster = None
        self._same_compute_on_resume = None
        self.discriminator = None
        if preferred_cluster is not None:
            self.preferred_cluster = preferred_cluster
        if same_compute_on_resume is not None:
            self.same_compute_on_resume = same_compute_on_resume

    @property
    def preferred_cluster(self) -> 'str':
        """Gets the preferred_cluster of this V1ProjectSettings.  # noqa: E501


        :return: The preferred_cluster of this V1ProjectSettings.  # noqa: E501
        :rtype: str
        """
        return self._preferred_cluster

    @preferred_cluster.setter
    def preferred_cluster(self, preferred_cluster: 'str'):
        """Sets the preferred_cluster of this V1ProjectSettings.


        :param preferred_cluster: The preferred_cluster of this V1ProjectSettings.  # noqa: E501
        :type: str
        """

        self._preferred_cluster = preferred_cluster

    @property
    def same_compute_on_resume(self) -> 'bool':
        """Gets the same_compute_on_resume of this V1ProjectSettings.  # noqa: E501


        :return: The same_compute_on_resume of this V1ProjectSettings.  # noqa: E501
        :rtype: bool
        """
        return self._same_compute_on_resume

    @same_compute_on_resume.setter
    def same_compute_on_resume(self, same_compute_on_resume: 'bool'):
        """Sets the same_compute_on_resume of this V1ProjectSettings.


        :param same_compute_on_resume: The same_compute_on_resume of this V1ProjectSettings.  # noqa: E501
        :type: bool
        """

        self._same_compute_on_resume = same_compute_on_resume

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
        if issubclass(V1ProjectSettings, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1ProjectSettings') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1ProjectSettings):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1ProjectSettings') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other