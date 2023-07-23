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


class V1CreateProjectRequest(object):
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
        'organization_id': 'str',
        'quotas': 'V1Quotas'
    }

    attribute_map = {
        'description': 'description',
        'display_name': 'displayName',
        'name': 'name',
        'organization_id': 'organizationId',
        'quotas': 'quotas'
    }

    def __init__(self,
                 description: 'str' = None,
                 display_name: 'str' = None,
                 name: 'str' = None,
                 organization_id: 'str' = None,
                 quotas: 'V1Quotas' = None):  # noqa: E501
        """V1CreateProjectRequest - a model defined in Swagger"""  # noqa: E501
        self._description = None
        self._display_name = None
        self._name = None
        self._organization_id = None
        self._quotas = None
        self.discriminator = None
        if description is not None:
            self.description = description
        if display_name is not None:
            self.display_name = display_name
        if name is not None:
            self.name = name
        if organization_id is not None:
            self.organization_id = organization_id
        if quotas is not None:
            self.quotas = quotas

    @property
    def description(self) -> 'str':
        """Gets the description of this V1CreateProjectRequest.  # noqa: E501


        :return: The description of this V1CreateProjectRequest.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: 'str'):
        """Sets the description of this V1CreateProjectRequest.


        :param description: The description of this V1CreateProjectRequest.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def display_name(self) -> 'str':
        """Gets the display_name of this V1CreateProjectRequest.  # noqa: E501


        :return: The display_name of this V1CreateProjectRequest.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: 'str'):
        """Sets the display_name of this V1CreateProjectRequest.


        :param display_name: The display_name of this V1CreateProjectRequest.  # noqa: E501
        :type: str
        """

        self._display_name = display_name

    @property
    def name(self) -> 'str':
        """Gets the name of this V1CreateProjectRequest.  # noqa: E501


        :return: The name of this V1CreateProjectRequest.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this V1CreateProjectRequest.


        :param name: The name of this V1CreateProjectRequest.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def organization_id(self) -> 'str':
        """Gets the organization_id of this V1CreateProjectRequest.  # noqa: E501


        :return: The organization_id of this V1CreateProjectRequest.  # noqa: E501
        :rtype: str
        """
        return self._organization_id

    @organization_id.setter
    def organization_id(self, organization_id: 'str'):
        """Sets the organization_id of this V1CreateProjectRequest.


        :param organization_id: The organization_id of this V1CreateProjectRequest.  # noqa: E501
        :type: str
        """

        self._organization_id = organization_id

    @property
    def quotas(self) -> 'V1Quotas':
        """Gets the quotas of this V1CreateProjectRequest.  # noqa: E501


        :return: The quotas of this V1CreateProjectRequest.  # noqa: E501
        :rtype: V1Quotas
        """
        return self._quotas

    @quotas.setter
    def quotas(self, quotas: 'V1Quotas'):
        """Sets the quotas of this V1CreateProjectRequest.


        :param quotas: The quotas of this V1CreateProjectRequest.  # noqa: E501
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
        if issubclass(V1CreateProjectRequest, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1CreateProjectRequest') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1CreateProjectRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1CreateProjectRequest') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
