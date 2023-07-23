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


class V1ListLightningworkResponse(object):
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
    swagger_types = {'lightningworks': 'list[Externalv1Lightningwork]'}

    attribute_map = {'lightningworks': 'lightningworks'}

    def __init__(self,
                 lightningworks: 'list[Externalv1Lightningwork]' = None
                 ):  # noqa: E501
        """V1ListLightningworkResponse - a model defined in Swagger"""  # noqa: E501
        self._lightningworks = None
        self.discriminator = None
        if lightningworks is not None:
            self.lightningworks = lightningworks

    @property
    def lightningworks(self) -> 'list[Externalv1Lightningwork]':
        """Gets the lightningworks of this V1ListLightningworkResponse.  # noqa: E501


        :return: The lightningworks of this V1ListLightningworkResponse.  # noqa: E501
        :rtype: list[Externalv1Lightningwork]
        """
        return self._lightningworks

    @lightningworks.setter
    def lightningworks(self, lightningworks: 'list[Externalv1Lightningwork]'):
        """Sets the lightningworks of this V1ListLightningworkResponse.


        :param lightningworks: The lightningworks of this V1ListLightningworkResponse.  # noqa: E501
        :type: list[Externalv1Lightningwork]
        """

        self._lightningworks = lightningworks

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
        if issubclass(V1ListLightningworkResponse, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1ListLightningworkResponse') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1ListLightningworkResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1ListLightningworkResponse') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
