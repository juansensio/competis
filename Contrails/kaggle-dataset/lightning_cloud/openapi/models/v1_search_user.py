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


class V1SearchUser(object):
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
        'auth_provider': 'str',
        'country': 'str',
        'first_name': 'str',
        'id': 'str',
        'last_name': 'str',
        'organization': 'str',
        'picture_url': 'str',
        'project_memberships': 'list[str]',
        'role': 'str',
        'username': 'str',
        'website': 'str'
    }

    attribute_map = {
        'auth_provider': 'authProvider',
        'country': 'country',
        'first_name': 'firstName',
        'id': 'id',
        'last_name': 'lastName',
        'organization': 'organization',
        'picture_url': 'pictureUrl',
        'project_memberships': 'projectMemberships',
        'role': 'role',
        'username': 'username',
        'website': 'website'
    }

    def __init__(self,
                 auth_provider: 'str' = None,
                 country: 'str' = None,
                 first_name: 'str' = None,
                 id: 'str' = None,
                 last_name: 'str' = None,
                 organization: 'str' = None,
                 picture_url: 'str' = None,
                 project_memberships: 'list[str]' = None,
                 role: 'str' = None,
                 username: 'str' = None,
                 website: 'str' = None):  # noqa: E501
        """V1SearchUser - a model defined in Swagger"""  # noqa: E501
        self._auth_provider = None
        self._country = None
        self._first_name = None
        self._id = None
        self._last_name = None
        self._organization = None
        self._picture_url = None
        self._project_memberships = None
        self._role = None
        self._username = None
        self._website = None
        self.discriminator = None
        if auth_provider is not None:
            self.auth_provider = auth_provider
        if country is not None:
            self.country = country
        if first_name is not None:
            self.first_name = first_name
        if id is not None:
            self.id = id
        if last_name is not None:
            self.last_name = last_name
        if organization is not None:
            self.organization = organization
        if picture_url is not None:
            self.picture_url = picture_url
        if project_memberships is not None:
            self.project_memberships = project_memberships
        if role is not None:
            self.role = role
        if username is not None:
            self.username = username
        if website is not None:
            self.website = website

    @property
    def auth_provider(self) -> 'str':
        """Gets the auth_provider of this V1SearchUser.  # noqa: E501


        :return: The auth_provider of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._auth_provider

    @auth_provider.setter
    def auth_provider(self, auth_provider: 'str'):
        """Sets the auth_provider of this V1SearchUser.


        :param auth_provider: The auth_provider of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._auth_provider = auth_provider

    @property
    def country(self) -> 'str':
        """Gets the country of this V1SearchUser.  # noqa: E501


        :return: The country of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._country

    @country.setter
    def country(self, country: 'str'):
        """Sets the country of this V1SearchUser.


        :param country: The country of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._country = country

    @property
    def first_name(self) -> 'str':
        """Gets the first_name of this V1SearchUser.  # noqa: E501


        :return: The first_name of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._first_name

    @first_name.setter
    def first_name(self, first_name: 'str'):
        """Sets the first_name of this V1SearchUser.


        :param first_name: The first_name of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._first_name = first_name

    @property
    def id(self) -> 'str':
        """Gets the id of this V1SearchUser.  # noqa: E501


        :return: The id of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: 'str'):
        """Sets the id of this V1SearchUser.


        :param id: The id of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def last_name(self) -> 'str':
        """Gets the last_name of this V1SearchUser.  # noqa: E501


        :return: The last_name of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._last_name

    @last_name.setter
    def last_name(self, last_name: 'str'):
        """Sets the last_name of this V1SearchUser.


        :param last_name: The last_name of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._last_name = last_name

    @property
    def organization(self) -> 'str':
        """Gets the organization of this V1SearchUser.  # noqa: E501


        :return: The organization of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._organization

    @organization.setter
    def organization(self, organization: 'str'):
        """Sets the organization of this V1SearchUser.


        :param organization: The organization of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._organization = organization

    @property
    def picture_url(self) -> 'str':
        """Gets the picture_url of this V1SearchUser.  # noqa: E501


        :return: The picture_url of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._picture_url

    @picture_url.setter
    def picture_url(self, picture_url: 'str'):
        """Sets the picture_url of this V1SearchUser.


        :param picture_url: The picture_url of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._picture_url = picture_url

    @property
    def project_memberships(self) -> 'list[str]':
        """Gets the project_memberships of this V1SearchUser.  # noqa: E501

        Project memberships are needed to fetch resources (like Lightning apps) that belong to this user.  # noqa: E501

        :return: The project_memberships of this V1SearchUser.  # noqa: E501
        :rtype: list[str]
        """
        return self._project_memberships

    @project_memberships.setter
    def project_memberships(self, project_memberships: 'list[str]'):
        """Sets the project_memberships of this V1SearchUser.

        Project memberships are needed to fetch resources (like Lightning apps) that belong to this user.  # noqa: E501

        :param project_memberships: The project_memberships of this V1SearchUser.  # noqa: E501
        :type: list[str]
        """

        self._project_memberships = project_memberships

    @property
    def role(self) -> 'str':
        """Gets the role of this V1SearchUser.  # noqa: E501


        :return: The role of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._role

    @role.setter
    def role(self, role: 'str'):
        """Sets the role of this V1SearchUser.


        :param role: The role of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._role = role

    @property
    def username(self) -> 'str':
        """Gets the username of this V1SearchUser.  # noqa: E501


        :return: The username of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._username

    @username.setter
    def username(self, username: 'str'):
        """Sets the username of this V1SearchUser.


        :param username: The username of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._username = username

    @property
    def website(self) -> 'str':
        """Gets the website of this V1SearchUser.  # noqa: E501


        :return: The website of this V1SearchUser.  # noqa: E501
        :rtype: str
        """
        return self._website

    @website.setter
    def website(self, website: 'str'):
        """Sets the website of this V1SearchUser.


        :param website: The website of this V1SearchUser.  # noqa: E501
        :type: str
        """

        self._website = website

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
        if issubclass(V1SearchUser, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1SearchUser') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1SearchUser):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1SearchUser') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other