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


class V1ClusterDriverStatus(object):
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
        'external': 'ProtobufAny',
        'kubernetes': 'V1KubernetesClusterStatus',
        'slurm': 'V1SlurmClusterStatus'
    }

    attribute_map = {
        'external': 'external',
        'kubernetes': 'kubernetes',
        'slurm': 'slurm'
    }

    def __init__(self,
                 external: 'ProtobufAny' = None,
                 kubernetes: 'V1KubernetesClusterStatus' = None,
                 slurm: 'V1SlurmClusterStatus' = None):  # noqa: E501
        """V1ClusterDriverStatus - a model defined in Swagger"""  # noqa: E501
        self._external = None
        self._kubernetes = None
        self._slurm = None
        self.discriminator = None
        if external is not None:
            self.external = external
        if kubernetes is not None:
            self.kubernetes = kubernetes
        if slurm is not None:
            self.slurm = slurm

    @property
    def external(self) -> 'ProtobufAny':
        """Gets the external of this V1ClusterDriverStatus.  # noqa: E501


        :return: The external of this V1ClusterDriverStatus.  # noqa: E501
        :rtype: ProtobufAny
        """
        return self._external

    @external.setter
    def external(self, external: 'ProtobufAny'):
        """Sets the external of this V1ClusterDriverStatus.


        :param external: The external of this V1ClusterDriverStatus.  # noqa: E501
        :type: ProtobufAny
        """

        self._external = external

    @property
    def kubernetes(self) -> 'V1KubernetesClusterStatus':
        """Gets the kubernetes of this V1ClusterDriverStatus.  # noqa: E501


        :return: The kubernetes of this V1ClusterDriverStatus.  # noqa: E501
        :rtype: V1KubernetesClusterStatus
        """
        return self._kubernetes

    @kubernetes.setter
    def kubernetes(self, kubernetes: 'V1KubernetesClusterStatus'):
        """Sets the kubernetes of this V1ClusterDriverStatus.


        :param kubernetes: The kubernetes of this V1ClusterDriverStatus.  # noqa: E501
        :type: V1KubernetesClusterStatus
        """

        self._kubernetes = kubernetes

    @property
    def slurm(self) -> 'V1SlurmClusterStatus':
        """Gets the slurm of this V1ClusterDriverStatus.  # noqa: E501


        :return: The slurm of this V1ClusterDriverStatus.  # noqa: E501
        :rtype: V1SlurmClusterStatus
        """
        return self._slurm

    @slurm.setter
    def slurm(self, slurm: 'V1SlurmClusterStatus'):
        """Sets the slurm of this V1ClusterDriverStatus.


        :param slurm: The slurm of this V1ClusterDriverStatus.  # noqa: E501
        :type: V1SlurmClusterStatus
        """

        self._slurm = slurm

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
        if issubclass(V1ClusterDriverStatus, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1ClusterDriverStatus') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1ClusterDriverStatus):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1ClusterDriverStatus') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
