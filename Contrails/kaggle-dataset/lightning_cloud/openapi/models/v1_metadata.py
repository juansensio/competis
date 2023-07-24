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


class V1Metadata(object):
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
        'annotations': 'dict(str, str)',
        'creation_timestamp': 'datetime',
        'deletion_timestamp': 'datetime',
        'display_name': 'str',
        'finalizers': 'list[str]',
        'id': 'str',
        'labels': 'dict(str, str)',
        'last_updated_by_user_at': 'datetime',
        'name': 'str',
        'project_id': 'str',
        'resource_version': 'str',
        'updated_at': 'datetime'
    }

    attribute_map = {
        'annotations': 'annotations',
        'creation_timestamp': 'creationTimestamp',
        'deletion_timestamp': 'deletionTimestamp',
        'display_name': 'displayName',
        'finalizers': 'finalizers',
        'id': 'id',
        'labels': 'labels',
        'last_updated_by_user_at': 'lastUpdatedByUserAt',
        'name': 'name',
        'project_id': 'projectId',
        'resource_version': 'resourceVersion',
        'updated_at': 'updatedAt'
    }

    def __init__(self,
                 annotations: 'dict(str, str)' = None,
                 creation_timestamp: 'datetime' = None,
                 deletion_timestamp: 'datetime' = None,
                 display_name: 'str' = None,
                 finalizers: 'list[str]' = None,
                 id: 'str' = None,
                 labels: 'dict(str, str)' = None,
                 last_updated_by_user_at: 'datetime' = None,
                 name: 'str' = None,
                 project_id: 'str' = None,
                 resource_version: 'str' = None,
                 updated_at: 'datetime' = None):  # noqa: E501
        """V1Metadata - a model defined in Swagger"""  # noqa: E501
        self._annotations = None
        self._creation_timestamp = None
        self._deletion_timestamp = None
        self._display_name = None
        self._finalizers = None
        self._id = None
        self._labels = None
        self._last_updated_by_user_at = None
        self._name = None
        self._project_id = None
        self._resource_version = None
        self._updated_at = None
        self.discriminator = None
        if annotations is not None:
            self.annotations = annotations
        if creation_timestamp is not None:
            self.creation_timestamp = creation_timestamp
        if deletion_timestamp is not None:
            self.deletion_timestamp = deletion_timestamp
        if display_name is not None:
            self.display_name = display_name
        if finalizers is not None:
            self.finalizers = finalizers
        if id is not None:
            self.id = id
        if labels is not None:
            self.labels = labels
        if last_updated_by_user_at is not None:
            self.last_updated_by_user_at = last_updated_by_user_at
        if name is not None:
            self.name = name
        if project_id is not None:
            self.project_id = project_id
        if resource_version is not None:
            self.resource_version = resource_version
        if updated_at is not None:
            self.updated_at = updated_at

    @property
    def annotations(self) -> 'dict(str, str)':
        """Gets the annotations of this V1Metadata.  # noqa: E501


        :return: The annotations of this V1Metadata.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: 'dict(str, str)'):
        """Sets the annotations of this V1Metadata.


        :param annotations: The annotations of this V1Metadata.  # noqa: E501
        :type: dict(str, str)
        """

        self._annotations = annotations

    @property
    def creation_timestamp(self) -> 'datetime':
        """Gets the creation_timestamp of this V1Metadata.  # noqa: E501


        :return: The creation_timestamp of this V1Metadata.  # noqa: E501
        :rtype: datetime
        """
        return self._creation_timestamp

    @creation_timestamp.setter
    def creation_timestamp(self, creation_timestamp: 'datetime'):
        """Sets the creation_timestamp of this V1Metadata.


        :param creation_timestamp: The creation_timestamp of this V1Metadata.  # noqa: E501
        :type: datetime
        """

        self._creation_timestamp = creation_timestamp

    @property
    def deletion_timestamp(self) -> 'datetime':
        """Gets the deletion_timestamp of this V1Metadata.  # noqa: E501


        :return: The deletion_timestamp of this V1Metadata.  # noqa: E501
        :rtype: datetime
        """
        return self._deletion_timestamp

    @deletion_timestamp.setter
    def deletion_timestamp(self, deletion_timestamp: 'datetime'):
        """Sets the deletion_timestamp of this V1Metadata.


        :param deletion_timestamp: The deletion_timestamp of this V1Metadata.  # noqa: E501
        :type: datetime
        """

        self._deletion_timestamp = deletion_timestamp

    @property
    def display_name(self) -> 'str':
        """Gets the display_name of this V1Metadata.  # noqa: E501


        :return: The display_name of this V1Metadata.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: 'str'):
        """Sets the display_name of this V1Metadata.


        :param display_name: The display_name of this V1Metadata.  # noqa: E501
        :type: str
        """

        self._display_name = display_name

    @property
    def finalizers(self) -> 'list[str]':
        """Gets the finalizers of this V1Metadata.  # noqa: E501


        :return: The finalizers of this V1Metadata.  # noqa: E501
        :rtype: list[str]
        """
        return self._finalizers

    @finalizers.setter
    def finalizers(self, finalizers: 'list[str]'):
        """Sets the finalizers of this V1Metadata.


        :param finalizers: The finalizers of this V1Metadata.  # noqa: E501
        :type: list[str]
        """

        self._finalizers = finalizers

    @property
    def id(self) -> 'str':
        """Gets the id of this V1Metadata.  # noqa: E501


        :return: The id of this V1Metadata.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: 'str'):
        """Sets the id of this V1Metadata.


        :param id: The id of this V1Metadata.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def labels(self) -> 'dict(str, str)':
        """Gets the labels of this V1Metadata.  # noqa: E501

        (Could) provide filtering options for list/watch requests, not implemented.  # noqa: E501

        :return: The labels of this V1Metadata.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._labels

    @labels.setter
    def labels(self, labels: 'dict(str, str)'):
        """Sets the labels of this V1Metadata.

        (Could) provide filtering options for list/watch requests, not implemented.  # noqa: E501

        :param labels: The labels of this V1Metadata.  # noqa: E501
        :type: dict(str, str)
        """

        self._labels = labels

    @property
    def last_updated_by_user_at(self) -> 'datetime':
        """Gets the last_updated_by_user_at of this V1Metadata.  # noqa: E501


        :return: The last_updated_by_user_at of this V1Metadata.  # noqa: E501
        :rtype: datetime
        """
        return self._last_updated_by_user_at

    @last_updated_by_user_at.setter
    def last_updated_by_user_at(self, last_updated_by_user_at: 'datetime'):
        """Sets the last_updated_by_user_at of this V1Metadata.


        :param last_updated_by_user_at: The last_updated_by_user_at of this V1Metadata.  # noqa: E501
        :type: datetime
        """

        self._last_updated_by_user_at = last_updated_by_user_at

    @property
    def name(self) -> 'str':
        """Gets the name of this V1Metadata.  # noqa: E501


        :return: The name of this V1Metadata.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this V1Metadata.


        :param name: The name of this V1Metadata.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def project_id(self) -> 'str':
        """Gets the project_id of this V1Metadata.  # noqa: E501


        :return: The project_id of this V1Metadata.  # noqa: E501
        :rtype: str
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id: 'str'):
        """Sets the project_id of this V1Metadata.


        :param project_id: The project_id of this V1Metadata.  # noqa: E501
        :type: str
        """

        self._project_id = project_id

    @property
    def resource_version(self) -> 'str':
        """Gets the resource_version of this V1Metadata.  # noqa: E501


        :return: The resource_version of this V1Metadata.  # noqa: E501
        :rtype: str
        """
        return self._resource_version

    @resource_version.setter
    def resource_version(self, resource_version: 'str'):
        """Sets the resource_version of this V1Metadata.


        :param resource_version: The resource_version of this V1Metadata.  # noqa: E501
        :type: str
        """

        self._resource_version = resource_version

    @property
    def updated_at(self) -> 'datetime':
        """Gets the updated_at of this V1Metadata.  # noqa: E501


        :return: The updated_at of this V1Metadata.  # noqa: E501
        :rtype: datetime
        """
        return self._updated_at

    @updated_at.setter
    def updated_at(self, updated_at: 'datetime'):
        """Sets the updated_at of this V1Metadata.


        :param updated_at: The updated_at of this V1Metadata.  # noqa: E501
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
        if issubclass(V1Metadata, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1Metadata') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1Metadata):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1Metadata') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other