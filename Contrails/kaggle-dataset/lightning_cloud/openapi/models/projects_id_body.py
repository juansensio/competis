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


class ProjectsIdBody(object):
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
        'description': 'str',
        'display_name': 'str',
        'name': 'str',
        'preferred_cluster': 'str',
        'quotas': 'V1Quotas'
    }

    attribute_map = {
        'description': 'description',
        'display_name': 'displayName',
        'name': 'name',
        'preferred_cluster': 'preferredCluster',
        'quotas': 'quotas'
    }

    def __init__(self,
                 description: 'str' = None,
                 display_name: 'str' = None,
                 name: 'str' = None,
                 preferred_cluster: 'str' = None,
                 quotas: 'V1Quotas' = None):  # noqa: E501
        """ProjectsIdBody - a model defined in Swagger"""  # noqa: E501
        self._description = None
        self._display_name = None
        self._name = None
        self._preferred_cluster = None
        self._quotas = None
        self.discriminator = None
        if description is not None:
            self.description = description
        if display_name is not None:
            self.display_name = display_name
        if name is not None:
            self.name = name
        if preferred_cluster is not None:
            self.preferred_cluster = preferred_cluster
        if quotas is not None:
            self.quotas = quotas

    @property
    def description(self) -> 'str':
        """Gets the description of this ProjectsIdBody.  # noqa: E501


        :return: The description of this ProjectsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: 'str'):
        """Sets the description of this ProjectsIdBody.


        :param description: The description of this ProjectsIdBody.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def display_name(self) -> 'str':
        """Gets the display_name of this ProjectsIdBody.  # noqa: E501


        :return: The display_name of this ProjectsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: 'str'):
        """Sets the display_name of this ProjectsIdBody.


        :param display_name: The display_name of this ProjectsIdBody.  # noqa: E501
        :type: str
        """

        self._display_name = display_name

    @property
    def name(self) -> 'str':
        """Gets the name of this ProjectsIdBody.  # noqa: E501


        :return: The name of this ProjectsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this ProjectsIdBody.


        :param name: The name of this ProjectsIdBody.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def preferred_cluster(self) -> 'str':
        """Gets the preferred_cluster of this ProjectsIdBody.  # noqa: E501


        :return: The preferred_cluster of this ProjectsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._preferred_cluster

    @preferred_cluster.setter
    def preferred_cluster(self, preferred_cluster: 'str'):
        """Sets the preferred_cluster of this ProjectsIdBody.


        :param preferred_cluster: The preferred_cluster of this ProjectsIdBody.  # noqa: E501
        :type: str
        """

        self._preferred_cluster = preferred_cluster

    @property
    def quotas(self) -> 'V1Quotas':
        """Gets the quotas of this ProjectsIdBody.  # noqa: E501


        :return: The quotas of this ProjectsIdBody.  # noqa: E501
        :rtype: V1Quotas
        """
        return self._quotas

    @quotas.setter
    def quotas(self, quotas: 'V1Quotas'):
        """Sets the quotas of this ProjectsIdBody.


        :param quotas: The quotas of this ProjectsIdBody.  # noqa: E501
        :type: V1Quotas
        """

        self._quotas = quotas

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
        if issubclass(ProjectsIdBody, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'ProjectsIdBody') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, ProjectsIdBody):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProjectsIdBody') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other