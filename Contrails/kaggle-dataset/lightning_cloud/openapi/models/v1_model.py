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


class V1Model(object):
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
        'categories': 'list[str]',
        'created_at': 'datetime',
        'description': 'str',
        'downloads': 'str',
        'id': 'str',
        'license': 'str',
        'metadata': 'dict(str, str)',
        'name': 'str',
        'private': 'bool',
        'project_id': 'str',
        'tags': 'list[str]',
        'updated_at': 'datetime'
    }

    attribute_map = {
        'categories': 'categories',
        'created_at': 'createdAt',
        'description': 'description',
        'downloads': 'downloads',
        'id': 'id',
        'license': 'license',
        'metadata': 'metadata',
        'name': 'name',
        'private': 'private',
        'project_id': 'projectId',
        'tags': 'tags',
        'updated_at': 'updatedAt'
    }

    def __init__(self,
                 categories: 'list[str]' = None,
                 created_at: 'datetime' = None,
                 description: 'str' = None,
                 downloads: 'str' = None,
                 id: 'str' = None,
                 license: 'str' = None,
                 metadata: 'dict(str, str)' = None,
                 name: 'str' = None,
                 private: 'bool' = None,
                 project_id: 'str' = None,
                 tags: 'list[str]' = None,
                 updated_at: 'datetime' = None):  # noqa: E501
        """V1Model - a model defined in Swagger"""  # noqa: E501
        self._categories = None
        self._created_at = None
        self._description = None
        self._downloads = None
        self._id = None
        self._license = None
        self._metadata = None
        self._name = None
        self._private = None
        self._project_id = None
        self._tags = None
        self._updated_at = None
        self.discriminator = None
        if categories is not None:
            self.categories = categories
        if created_at is not None:
            self.created_at = created_at
        if description is not None:
            self.description = description
        if downloads is not None:
            self.downloads = downloads
        if id is not None:
            self.id = id
        if license is not None:
            self.license = license
        if metadata is not None:
            self.metadata = metadata
        if name is not None:
            self.name = name
        if private is not None:
            self.private = private
        if project_id is not None:
            self.project_id = project_id
        if tags is not None:
            self.tags = tags
        if updated_at is not None:
            self.updated_at = updated_at

    @property
    def categories(self) -> 'list[str]':
        """Gets the categories of this V1Model.  # noqa: E501


        :return: The categories of this V1Model.  # noqa: E501
        :rtype: list[str]
        """
        return self._categories

    @categories.setter
    def categories(self, categories: 'list[str]'):
        """Sets the categories of this V1Model.


        :param categories: The categories of this V1Model.  # noqa: E501
        :type: list[str]
        """

        self._categories = categories

    @property
    def created_at(self) -> 'datetime':
        """Gets the created_at of this V1Model.  # noqa: E501


        :return: The created_at of this V1Model.  # noqa: E501
        :rtype: datetime
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at: 'datetime'):
        """Sets the created_at of this V1Model.


        :param created_at: The created_at of this V1Model.  # noqa: E501
        :type: datetime
        """

        self._created_at = created_at

    @property
    def description(self) -> 'str':
        """Gets the description of this V1Model.  # noqa: E501


        :return: The description of this V1Model.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: 'str'):
        """Sets the description of this V1Model.


        :param description: The description of this V1Model.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def downloads(self) -> 'str':
        """Gets the downloads of this V1Model.  # noqa: E501


        :return: The downloads of this V1Model.  # noqa: E501
        :rtype: str
        """
        return self._downloads

    @downloads.setter
    def downloads(self, downloads: 'str'):
        """Sets the downloads of this V1Model.


        :param downloads: The downloads of this V1Model.  # noqa: E501
        :type: str
        """

        self._downloads = downloads

    @property
    def id(self) -> 'str':
        """Gets the id of this V1Model.  # noqa: E501


        :return: The id of this V1Model.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: 'str'):
        """Sets the id of this V1Model.


        :param id: The id of this V1Model.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def license(self) -> 'str':
        """Gets the license of this V1Model.  # noqa: E501


        :return: The license of this V1Model.  # noqa: E501
        :rtype: str
        """
        return self._license

    @license.setter
    def license(self, license: 'str'):
        """Sets the license of this V1Model.


        :param license: The license of this V1Model.  # noqa: E501
        :type: str
        """

        self._license = license

    @property
    def metadata(self) -> 'dict(str, str)':
        """Gets the metadata of this V1Model.  # noqa: E501


        :return: The metadata of this V1Model.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: 'dict(str, str)'):
        """Sets the metadata of this V1Model.


        :param metadata: The metadata of this V1Model.  # noqa: E501
        :type: dict(str, str)
        """

        self._metadata = metadata

    @property
    def name(self) -> 'str':
        """Gets the name of this V1Model.  # noqa: E501


        :return: The name of this V1Model.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this V1Model.


        :param name: The name of this V1Model.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def private(self) -> 'bool':
        """Gets the private of this V1Model.  # noqa: E501


        :return: The private of this V1Model.  # noqa: E501
        :rtype: bool
        """
        return self._private

    @private.setter
    def private(self, private: 'bool'):
        """Sets the private of this V1Model.


        :param private: The private of this V1Model.  # noqa: E501
        :type: bool
        """

        self._private = private

    @property
    def project_id(self) -> 'str':
        """Gets the project_id of this V1Model.  # noqa: E501


        :return: The project_id of this V1Model.  # noqa: E501
        :rtype: str
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id: 'str'):
        """Sets the project_id of this V1Model.


        :param project_id: The project_id of this V1Model.  # noqa: E501
        :type: str
        """

        self._project_id = project_id

    @property
    def tags(self) -> 'list[str]':
        """Gets the tags of this V1Model.  # noqa: E501


        :return: The tags of this V1Model.  # noqa: E501
        :rtype: list[str]
        """
        return self._tags

    @tags.setter
    def tags(self, tags: 'list[str]'):
        """Sets the tags of this V1Model.


        :param tags: The tags of this V1Model.  # noqa: E501
        :type: list[str]
        """

        self._tags = tags

    @property
    def updated_at(self) -> 'datetime':
        """Gets the updated_at of this V1Model.  # noqa: E501


        :return: The updated_at of this V1Model.  # noqa: E501
        :rtype: datetime
        """
        return self._updated_at

    @updated_at.setter
    def updated_at(self, updated_at: 'datetime'):
        """Sets the updated_at of this V1Model.


        :param updated_at: The updated_at of this V1Model.  # noqa: E501
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
        if issubclass(V1Model, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1Model') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1Model):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1Model') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
