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


class V1ClusterAccelerator(object):
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
        'accelerator_type': 'str',
        'cost': 'float',
        'device_card': 'str',
        'device_info': 'str',
        'display_name': 'str',
        'enabled': 'bool',
        'instance_id': 'str',
        'resources': 'V1Resources',
        'slug': 'str'
    }

    attribute_map = {
        'accelerator_type': 'acceleratorType',
        'cost': 'cost',
        'device_card': 'deviceCard',
        'device_info': 'deviceInfo',
        'display_name': 'displayName',
        'enabled': 'enabled',
        'instance_id': 'instanceId',
        'resources': 'resources',
        'slug': 'slug'
    }

    def __init__(self,
                 accelerator_type: 'str' = None,
                 cost: 'float' = None,
                 device_card: 'str' = None,
                 device_info: 'str' = None,
                 display_name: 'str' = None,
                 enabled: 'bool' = None,
                 instance_id: 'str' = None,
                 resources: 'V1Resources' = None,
                 slug: 'str' = None):  # noqa: E501
        """V1ClusterAccelerator - a model defined in Swagger"""  # noqa: E501
        self._accelerator_type = None
        self._cost = None
        self._device_card = None
        self._device_info = None
        self._display_name = None
        self._enabled = None
        self._instance_id = None
        self._resources = None
        self._slug = None
        self.discriminator = None
        if accelerator_type is not None:
            self.accelerator_type = accelerator_type
        if cost is not None:
            self.cost = cost
        if device_card is not None:
            self.device_card = device_card
        if device_info is not None:
            self.device_info = device_info
        if display_name is not None:
            self.display_name = display_name
        if enabled is not None:
            self.enabled = enabled
        if instance_id is not None:
            self.instance_id = instance_id
        if resources is not None:
            self.resources = resources
        if slug is not None:
            self.slug = slug

    @property
    def accelerator_type(self) -> 'str':
        """Gets the accelerator_type of this V1ClusterAccelerator.  # noqa: E501


        :return: The accelerator_type of this V1ClusterAccelerator.  # noqa: E501
        :rtype: str
        """
        return self._accelerator_type

    @accelerator_type.setter
    def accelerator_type(self, accelerator_type: 'str'):
        """Sets the accelerator_type of this V1ClusterAccelerator.


        :param accelerator_type: The accelerator_type of this V1ClusterAccelerator.  # noqa: E501
        :type: str
        """

        self._accelerator_type = accelerator_type

    @property
    def cost(self) -> 'float':
        """Gets the cost of this V1ClusterAccelerator.  # noqa: E501


        :return: The cost of this V1ClusterAccelerator.  # noqa: E501
        :rtype: float
        """
        return self._cost

    @cost.setter
    def cost(self, cost: 'float'):
        """Sets the cost of this V1ClusterAccelerator.


        :param cost: The cost of this V1ClusterAccelerator.  # noqa: E501
        :type: float
        """

        self._cost = cost

    @property
    def device_card(self) -> 'str':
        """Gets the device_card of this V1ClusterAccelerator.  # noqa: E501


        :return: The device_card of this V1ClusterAccelerator.  # noqa: E501
        :rtype: str
        """
        return self._device_card

    @device_card.setter
    def device_card(self, device_card: 'str'):
        """Sets the device_card of this V1ClusterAccelerator.


        :param device_card: The device_card of this V1ClusterAccelerator.  # noqa: E501
        :type: str
        """

        self._device_card = device_card

    @property
    def device_info(self) -> 'str':
        """Gets the device_info of this V1ClusterAccelerator.  # noqa: E501


        :return: The device_info of this V1ClusterAccelerator.  # noqa: E501
        :rtype: str
        """
        return self._device_info

    @device_info.setter
    def device_info(self, device_info: 'str'):
        """Sets the device_info of this V1ClusterAccelerator.


        :param device_info: The device_info of this V1ClusterAccelerator.  # noqa: E501
        :type: str
        """

        self._device_info = device_info

    @property
    def display_name(self) -> 'str':
        """Gets the display_name of this V1ClusterAccelerator.  # noqa: E501


        :return: The display_name of this V1ClusterAccelerator.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name: 'str'):
        """Sets the display_name of this V1ClusterAccelerator.


        :param display_name: The display_name of this V1ClusterAccelerator.  # noqa: E501
        :type: str
        """

        self._display_name = display_name

    @property
    def enabled(self) -> 'bool':
        """Gets the enabled of this V1ClusterAccelerator.  # noqa: E501


        :return: The enabled of this V1ClusterAccelerator.  # noqa: E501
        :rtype: bool
        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: 'bool'):
        """Sets the enabled of this V1ClusterAccelerator.


        :param enabled: The enabled of this V1ClusterAccelerator.  # noqa: E501
        :type: bool
        """

        self._enabled = enabled

    @property
    def instance_id(self) -> 'str':
        """Gets the instance_id of this V1ClusterAccelerator.  # noqa: E501


        :return: The instance_id of this V1ClusterAccelerator.  # noqa: E501
        :rtype: str
        """
        return self._instance_id

    @instance_id.setter
    def instance_id(self, instance_id: 'str'):
        """Sets the instance_id of this V1ClusterAccelerator.


        :param instance_id: The instance_id of this V1ClusterAccelerator.  # noqa: E501
        :type: str
        """

        self._instance_id = instance_id

    @property
    def resources(self) -> 'V1Resources':
        """Gets the resources of this V1ClusterAccelerator.  # noqa: E501


        :return: The resources of this V1ClusterAccelerator.  # noqa: E501
        :rtype: V1Resources
        """
        return self._resources

    @resources.setter
    def resources(self, resources: 'V1Resources'):
        """Sets the resources of this V1ClusterAccelerator.


        :param resources: The resources of this V1ClusterAccelerator.  # noqa: E501
        :type: V1Resources
        """

        self._resources = resources

    @property
    def slug(self) -> 'str':
        """Gets the slug of this V1ClusterAccelerator.  # noqa: E501


        :return: The slug of this V1ClusterAccelerator.  # noqa: E501
        :rtype: str
        """
        return self._slug

    @slug.setter
    def slug(self, slug: 'str'):
        """Sets the slug of this V1ClusterAccelerator.


        :param slug: The slug of this V1ClusterAccelerator.  # noqa: E501
        :type: str
        """

        self._slug = slug

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
        if issubclass(V1ClusterAccelerator, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1ClusterAccelerator') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1ClusterAccelerator):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1ClusterAccelerator') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other