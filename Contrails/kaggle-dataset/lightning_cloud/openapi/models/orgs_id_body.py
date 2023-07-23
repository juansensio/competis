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


class OrgsIdBody(object):
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
        'domain': 'str',
        'email': 'str',
        'location': 'str',
        'twitter_username': 'str'
    }

    attribute_map = {
        'description': 'description',
        'display_name': 'displayName',
        'domain': 'domain',
        'email': 'email',
        'location': 'location',
        'twitter_username': 'twitterUsername'
    }

    def __init__(self,
                 description: 'str' = None,
                 display_name: 'str' = None,
                 domain: 'str' = None,
                 email: 'str' = None,
                 location: 'str' = None,
                 twitter_username: 'str' = None):  # noqa: E501
        """OrgsIdBody - a model defined in Swagger"""  # noqa: E501
        self._description = None
        self._display_name = None
        self._domain = None
        self._email = None
        self._location = None
        self._twitter_username = None
        self.discriminator = None
        if description is not None:
            self.description = description
        if display_name is not None:
            self.display_name = display_name
        if domain is not None:
            self.domain = domain
        if email is not None:
            self.email = email
        if location is not None:
            self.location = location
        if twitter_username is not None:
            self.twitter_username = twitter_username

    @property
    def description(self) -> 'str':
        """Gets the description of this OrgsIdBody.  # noqa: E501


        :return: The description of this OrgsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: 'str'):
        """Sets the description of this OrgsIdBody.


        :param description: The description of this OrgsIdBody.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def display_name(self) -> 'str':
        """Gets the display_name of this OrgsIdBody.  # noqa: E501


        :return: The display_name of this OrgsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: 'str'):
        """Sets the display_name of this OrgsIdBody.


        :param display_name: The display_name of this OrgsIdBody.  # noqa: E501
        :type: str
        """

        self._display_name = display_name

    @property
    def domain(self) -> 'str':
        """Gets the domain of this OrgsIdBody.  # noqa: E501


        :return: The domain of this OrgsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._domain

    @domain.setter
    def domain(self, domain: 'str'):
        """Sets the domain of this OrgsIdBody.


        :param domain: The domain of this OrgsIdBody.  # noqa: E501
        :type: str
        """

        self._domain = domain

    @property
    def email(self) -> 'str':
        """Gets the email of this OrgsIdBody.  # noqa: E501


        :return: The email of this OrgsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._email

    @email.setter
    def email(self, email: 'str'):
        """Sets the email of this OrgsIdBody.


        :param email: The email of this OrgsIdBody.  # noqa: E501
        :type: str
        """

        self._email = email

    @property
    def location(self) -> 'str':
        """Gets the location of this OrgsIdBody.  # noqa: E501


        :return: The location of this OrgsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._location

    @location.setter
    def location(self, location: 'str'):
        """Sets the location of this OrgsIdBody.


        :param location: The location of this OrgsIdBody.  # noqa: E501
        :type: str
        """

        self._location = location

    @property
    def twitter_username(self) -> 'str':
        """Gets the twitter_username of this OrgsIdBody.  # noqa: E501


        :return: The twitter_username of this OrgsIdBody.  # noqa: E501
        :rtype: str
        """
        return self._twitter_username

    @twitter_username.setter
    def twitter_username(self, twitter_username: 'str'):
        """Sets the twitter_username of this OrgsIdBody.


        :param twitter_username: The twitter_username of this OrgsIdBody.  # noqa: E501
        :type: str
        """

        self._twitter_username = twitter_username

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
        if issubclass(OrgsIdBody, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'OrgsIdBody') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, OrgsIdBody):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OrgsIdBody') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
