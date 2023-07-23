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


class V1PublishedCloudSpaceResponse(object):
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
        'creation_timestamp': 'datetime',
        'description': 'str',
        'display_name': 'str',
        'engagement_counts': 'dict(str, str)',
        'id': 'str',
        'name': 'str',
        'project_id': 'str',
        'project_name': 'str',
        'project_owner_name': 'str',
        'published_at': 'datetime',
        'studio_creator_username': 'str',
        'thumbnail_url': 'str'
    }

    attribute_map = {
        'creation_timestamp': 'creationTimestamp',
        'description': 'description',
        'display_name': 'displayName',
        'engagement_counts': 'engagementCounts',
        'id': 'id',
        'name': 'name',
        'project_id': 'projectId',
        'project_name': 'projectName',
        'project_owner_name': 'projectOwnerName',
        'published_at': 'publishedAt',
        'studio_creator_username': 'studioCreatorUsername',
        'thumbnail_url': 'thumbnailUrl'
    }

    def __init__(self,
                 creation_timestamp: 'datetime' = None,
                 description: 'str' = None,
                 display_name: 'str' = None,
                 engagement_counts: 'dict(str, str)' = None,
                 id: 'str' = None,
                 name: 'str' = None,
                 project_id: 'str' = None,
                 project_name: 'str' = None,
                 project_owner_name: 'str' = None,
                 published_at: 'datetime' = None,
                 studio_creator_username: 'str' = None,
                 thumbnail_url: 'str' = None):  # noqa: E501
        """V1PublishedCloudSpaceResponse - a model defined in Swagger"""  # noqa: E501
        self._creation_timestamp = None
        self._description = None
        self._display_name = None
        self._engagement_counts = None
        self._id = None
        self._name = None
        self._project_id = None
        self._project_name = None
        self._project_owner_name = None
        self._published_at = None
        self._studio_creator_username = None
        self._thumbnail_url = None
        self.discriminator = None
        if creation_timestamp is not None:
            self.creation_timestamp = creation_timestamp
        if description is not None:
            self.description = description
        if display_name is not None:
            self.display_name = display_name
        if engagement_counts is not None:
            self.engagement_counts = engagement_counts
        if id is not None:
            self.id = id
        if name is not None:
            self.name = name
        if project_id is not None:
            self.project_id = project_id
        if project_name is not None:
            self.project_name = project_name
        if project_owner_name is not None:
            self.project_owner_name = project_owner_name
        if published_at is not None:
            self.published_at = published_at
        if studio_creator_username is not None:
            self.studio_creator_username = studio_creator_username
        if thumbnail_url is not None:
            self.thumbnail_url = thumbnail_url

    @property
    def creation_timestamp(self) -> 'datetime':
        """Gets the creation_timestamp of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The creation_timestamp of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: datetime
        """
        return self._creation_timestamp

    @creation_timestamp.setter
    def creation_timestamp(self, creation_timestamp: 'datetime'):
        """Sets the creation_timestamp of this V1PublishedCloudSpaceResponse.


        :param creation_timestamp: The creation_timestamp of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: datetime
        """

        self._creation_timestamp = creation_timestamp

    @property
    def description(self) -> 'str':
        """Gets the description of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The description of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: 'str'):
        """Sets the description of this V1PublishedCloudSpaceResponse.


        :param description: The description of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def display_name(self) -> 'str':
        """Gets the display_name of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The display_name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: 'str'):
        """Sets the display_name of this V1PublishedCloudSpaceResponse.


        :param display_name: The display_name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._display_name = display_name

    @property
    def engagement_counts(self) -> 'dict(str, str)':
        """Gets the engagement_counts of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The engagement_counts of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._engagement_counts

    @engagement_counts.setter
    def engagement_counts(self, engagement_counts: 'dict(str, str)'):
        """Sets the engagement_counts of this V1PublishedCloudSpaceResponse.


        :param engagement_counts: The engagement_counts of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: dict(str, str)
        """

        self._engagement_counts = engagement_counts

    @property
    def id(self) -> 'str':
        """Gets the id of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The id of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: 'str'):
        """Sets the id of this V1PublishedCloudSpaceResponse.


        :param id: The id of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def name(self) -> 'str':
        """Gets the name of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this V1PublishedCloudSpaceResponse.


        :param name: The name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def project_id(self) -> 'str':
        """Gets the project_id of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The project_id of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id: 'str'):
        """Sets the project_id of this V1PublishedCloudSpaceResponse.


        :param project_id: The project_id of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._project_id = project_id

    @property
    def project_name(self) -> 'str':
        """Gets the project_name of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The project_name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._project_name

    @project_name.setter
    def project_name(self, project_name: 'str'):
        """Sets the project_name of this V1PublishedCloudSpaceResponse.


        :param project_name: The project_name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._project_name = project_name

    @property
    def project_owner_name(self) -> 'str':
        """Gets the project_owner_name of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The project_owner_name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._project_owner_name

    @project_owner_name.setter
    def project_owner_name(self, project_owner_name: 'str'):
        """Sets the project_owner_name of this V1PublishedCloudSpaceResponse.


        :param project_owner_name: The project_owner_name of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._project_owner_name = project_owner_name

    @property
    def published_at(self) -> 'datetime':
        """Gets the published_at of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The published_at of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: datetime
        """
        return self._published_at

    @published_at.setter
    def published_at(self, published_at: 'datetime'):
        """Sets the published_at of this V1PublishedCloudSpaceResponse.


        :param published_at: The published_at of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: datetime
        """

        self._published_at = published_at

    @property
    def studio_creator_username(self) -> 'str':
        """Gets the studio_creator_username of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The studio_creator_username of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._studio_creator_username

    @studio_creator_username.setter
    def studio_creator_username(self, studio_creator_username: 'str'):
        """Sets the studio_creator_username of this V1PublishedCloudSpaceResponse.


        :param studio_creator_username: The studio_creator_username of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._studio_creator_username = studio_creator_username

    @property
    def thumbnail_url(self) -> 'str':
        """Gets the thumbnail_url of this V1PublishedCloudSpaceResponse.  # noqa: E501


        :return: The thumbnail_url of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :rtype: str
        """
        return self._thumbnail_url

    @thumbnail_url.setter
    def thumbnail_url(self, thumbnail_url: 'str'):
        """Sets the thumbnail_url of this V1PublishedCloudSpaceResponse.


        :param thumbnail_url: The thumbnail_url of this V1PublishedCloudSpaceResponse.  # noqa: E501
        :type: str
        """

        self._thumbnail_url = thumbnail_url

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
        if issubclass(V1PublishedCloudSpaceResponse, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1PublishedCloudSpaceResponse') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1PublishedCloudSpaceResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1PublishedCloudSpaceResponse') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
