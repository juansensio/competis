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


class V1LightningworkEvent(object):
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
        'desired_state': 'V1LightningworkState',
        'id': 'str',
        'message': 'str',
        'phase': 'V1LightningworkState',
        'timestamp': 'datetime',
        'total_cost': 'float'
    }

    attribute_map = {
        'desired_state': 'desiredState',
        'id': 'id',
        'message': 'message',
        'phase': 'phase',
        'timestamp': 'timestamp',
        'total_cost': 'totalCost'
    }

    def __init__(self,
                 desired_state: 'V1LightningworkState' = None,
                 id: 'str' = None,
                 message: 'str' = None,
                 phase: 'V1LightningworkState' = None,
                 timestamp: 'datetime' = None,
                 total_cost: 'float' = None):  # noqa: E501
        """V1LightningworkEvent - a model defined in Swagger"""  # noqa: E501
        self._desired_state = None
        self._id = None
        self._message = None
        self._phase = None
        self._timestamp = None
        self._total_cost = None
        self.discriminator = None
        if desired_state is not None:
            self.desired_state = desired_state
        if id is not None:
            self.id = id
        if message is not None:
            self.message = message
        if phase is not None:
            self.phase = phase
        if timestamp is not None:
            self.timestamp = timestamp
        if total_cost is not None:
            self.total_cost = total_cost

    @property
    def desired_state(self) -> 'V1LightningworkState':
        """Gets the desired_state of this V1LightningworkEvent.  # noqa: E501


        :return: The desired_state of this V1LightningworkEvent.  # noqa: E501
        :rtype: V1LightningworkState
        """
        return self._desired_state

    @desired_state.setter
    def desired_state(self, desired_state: 'V1LightningworkState'):
        """Sets the desired_state of this V1LightningworkEvent.


        :param desired_state: The desired_state of this V1LightningworkEvent.  # noqa: E501
        :type: V1LightningworkState
        """

        self._desired_state = desired_state

    @property
    def id(self) -> 'str':
        """Gets the id of this V1LightningworkEvent.  # noqa: E501


        :return: The id of this V1LightningworkEvent.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: 'str'):
        """Sets the id of this V1LightningworkEvent.


        :param id: The id of this V1LightningworkEvent.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def message(self) -> 'str':
        """Gets the message of this V1LightningworkEvent.  # noqa: E501


        :return: The message of this V1LightningworkEvent.  # noqa: E501
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message: 'str'):
        """Sets the message of this V1LightningworkEvent.


        :param message: The message of this V1LightningworkEvent.  # noqa: E501
        :type: str
        """

        self._message = message

    @property
    def phase(self) -> 'V1LightningworkState':
        """Gets the phase of this V1LightningworkEvent.  # noqa: E501


        :return: The phase of this V1LightningworkEvent.  # noqa: E501
        :rtype: V1LightningworkState
        """
        return self._phase

    @phase.setter
    def phase(self, phase: 'V1LightningworkState'):
        """Sets the phase of this V1LightningworkEvent.


        :param phase: The phase of this V1LightningworkEvent.  # noqa: E501
        :type: V1LightningworkState
        """

        self._phase = phase

    @property
    def timestamp(self) -> 'datetime':
        """Gets the timestamp of this V1LightningworkEvent.  # noqa: E501


        :return: The timestamp of this V1LightningworkEvent.  # noqa: E501
        :rtype: datetime
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp: 'datetime'):
        """Sets the timestamp of this V1LightningworkEvent.


        :param timestamp: The timestamp of this V1LightningworkEvent.  # noqa: E501
        :type: datetime
        """

        self._timestamp = timestamp

    @property
    def total_cost(self) -> 'float':
        """Gets the total_cost of this V1LightningworkEvent.  # noqa: E501


        :return: The total_cost of this V1LightningworkEvent.  # noqa: E501
        :rtype: float
        """
        return self._total_cost

    @total_cost.setter
    def total_cost(self, total_cost: 'float'):
        """Sets the total_cost of this V1LightningworkEvent.


        :param total_cost: The total_cost of this V1LightningworkEvent.  # noqa: E501
        :type: float
        """

        self._total_cost = total_cost

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
        if issubclass(V1LightningworkEvent, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1LightningworkEvent') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1LightningworkEvent):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1LightningworkEvent') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other