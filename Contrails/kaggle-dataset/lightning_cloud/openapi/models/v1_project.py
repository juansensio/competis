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


class V1Project(object):
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
        'created_at': 'datetime',
        'description': 'str',
        'display_name': 'str',
        'id': 'str',
        'name': 'str',
        'owner_id': 'str',
        'owner_type': 'V1OwnerType',
        'private': 'bool',
        'project_settings': 'V1ProjectSettings',
        'quotas': 'V1Quotas',
        'updated_at': 'datetime'
    }

    attribute_map = {
        'created_at': 'createdAt',
        'description': 'description',
        'display_name': 'displayName',
        'id': 'id',
        'name': 'name',
        'owner_id': 'ownerId',
        'owner_type': 'ownerType',
        'private': 'private',
        'project_settings': 'projectSettings',
        'quotas': 'quotas',
        'updated_at': 'updatedAt'
    }

    def __init__(self,
                 created_at: 'datetime' = None,
                 description: 'str' = None,
                 display_name: 'str' = None,
                 id: 'str' = None,
                 name: 'str' = None,
                 owner_id: 'str' = None,
                 owner_type: 'V1OwnerType' = None,
                 private: 'bool' = None,
                 project_settings: 'V1ProjectSettings' = None,
                 quotas: 'V1Quotas' = None,
                 updated_at: 'datetime' = None):  # noqa: E501
        """V1Project - a model defined in Swagger"""  # noqa: E501
        self._created_at = None
        self._description = None
        self._display_name = None
        self._id = None
        self._name = None
        self._owner_id = None
        self._owner_type = None
        self._private = None
        self._project_settings = None
        self._quotas = None
        self._updated_at = None
        self.discriminator = None
        if created_at is not None:
            self.created_at = created_at
        if description is not None:
            self.description = description
        if display_name is not None:
            self.display_name = display_name
        if id is not None:
            self.id = id
        if name is not None:
            self.name = name
        if owner_id is not None:
            self.owner_id = owner_id
        if owner_type is not None:
            self.owner_type = owner_type
        if private is not None:
            self.private = private
        if project_settings is not None:
            self.project_settings = project_settings
        if quotas is not None:
            self.quotas = quotas
        if updated_at is not None:
            self.updated_at = updated_at

    @property
    def created_at(self) -> 'datetime':
        """Gets the created_at of this V1Project.  # noqa: E501


        :return: The created_at of this V1Project.  # noqa: E501
        :rtype: datetime
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at: 'datetime'):
        """Sets the created_at of this V1Project.


        :param created_at: The created_at of this V1Project.  # noqa: E501
        :type: datetime
        """

        self._created_at = created_at

    @property
    def description(self) -> 'str':
        """Gets the description of this V1Project.  # noqa: E501


        :return: The description of this V1Project.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: 'str'):
        """Sets the description of this V1Project.


        :param description: The description of this V1Project.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def display_name(self) -> 'str':
        """Gets the display_name of this V1Project.  # noqa: E501


        :return: The display_name of this V1Project.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: 'str'):
        """Sets the display_name of this V1Project.


        :param display_name: The display_name of this V1Project.  # noqa: E501
        :type: str
        """

        self._display_name = display_name

    @property
    def id(self) -> 'str':
        """Gets the id of this V1Project.  # noqa: E501


        :return: The id of this V1Project.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: 'str'):
        """Sets the id of this V1Project.


        :param id: The id of this V1Project.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def name(self) -> 'str':
        """Gets the name of this V1Project.  # noqa: E501


        :return: The name of this V1Project.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this V1Project.


        :param name: The name of this V1Project.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def owner_id(self) -> 'str':
        """Gets the owner_id of this V1Project.  # noqa: E501


        :return: The owner_id of this V1Project.  # noqa: E501
        :rtype: str
        """
        return self._owner_id

    @owner_id.setter
    def owner_id(self, owner_id: 'str'):
        """Sets the owner_id of this V1Project.


        :param owner_id: The owner_id of this V1Project.  # noqa: E501
        :type: str
        """

        self._owner_id = owner_id

    @property
    def owner_type(self) -> 'V1OwnerType':
        """Gets the owner_type of this V1Project.  # noqa: E501


        :return: The owner_type of this V1Project.  # noqa: E501
        :rtype: V1OwnerType
        """
        return self._owner_type

    @owner_type.setter
    def owner_type(self, owner_type: 'V1OwnerType'):
        """Sets the owner_type of this V1Project.


        :param owner_type: The owner_type of this V1Project.  # noqa: E501
        :type: V1OwnerType
        """

        self._owner_type = owner_type

    @property
    def private(self) -> 'bool':
        """Gets the private of this V1Project.  # noqa: E501


        :return: The private of this V1Project.  # noqa: E501
        :rtype: bool
        """
        return self._private

    @private.setter
    def private(self, private: 'bool'):
        """Sets the private of this V1Project.


        :param private: The private of this V1Project.  # noqa: E501
        :type: bool
        """

        self._private = private

    @property
    def project_settings(self) -> 'V1ProjectSettings':
        """Gets the project_settings of this V1Project.  # noqa: E501


        :return: The project_settings of this V1Project.  # noqa: E501
        :rtype: V1ProjectSettings
        """
        return self._project_settings

    @project_settings.setter
    def project_settings(self, project_settings: 'V1ProjectSettings'):
        """Sets the project_settings of this V1Project.


        :param project_settings: The project_settings of this V1Project.  # noqa: E501
        :type: V1ProjectSettings
        """

        self._project_settings = project_settings

    @property
    def quotas(self) -> 'V1Quotas':
        """Gets the quotas of this V1Project.  # noqa: E501


        :return: The quotas of this V1Project.  # noqa: E501
        :rtype: V1Quotas
        """
        return self._quotas

    @quotas.setter
    def quotas(self, quotas: 'V1Quotas'):
        """Sets the quotas of this V1Project.


        :param quotas: The quotas of this V1Project.  # noqa: E501
        :type: V1Quotas
        """

        self._quotas = quotas

    @property
    def updated_at(self) -> 'datetime':
        """Gets the updated_at of this V1Project.  # noqa: E501


        :return: The updated_at of this V1Project.  # noqa: E501
        :rtype: datetime
        """
        return self._updated_at

    @updated_at.setter
    def updated_at(self, updated_at: 'datetime'):
        """Sets the updated_at of this V1Project.


        :param updated_at: The updated_at of this V1Project.  # noqa: E501
        :type: datetime
        """

        self._updated_at = updated_at

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
        if issubclass(V1Project, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1Project') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1Project):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1Project') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
