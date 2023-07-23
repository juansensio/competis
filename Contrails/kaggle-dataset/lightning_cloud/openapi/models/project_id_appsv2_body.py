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


class ProjectIdAppsv2Body(object):
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
        'can_download_source_code': 'bool',
        'cloud_space_instance_cpu_image_override': 'str',
        'cloud_space_instance_gpu_image_override': 'str',
        'name': 'str'
    }

    attribute_map = {
        'can_download_source_code': 'canDownloadSourceCode',
        'cloud_space_instance_cpu_image_override':
        'cloudSpaceInstanceCpuImageOverride',
        'cloud_space_instance_gpu_image_override':
        'cloudSpaceInstanceGpuImageOverride',
        'name': 'name'
    }

    def __init__(self,
                 can_download_source_code: 'bool' = None,
                 cloud_space_instance_cpu_image_override: 'str' = None,
                 cloud_space_instance_gpu_image_override: 'str' = None,
                 name: 'str' = None):  # noqa: E501
        """ProjectIdAppsv2Body - a model defined in Swagger"""  # noqa: E501
        self._can_download_source_code = None
        self._cloud_space_instance_cpu_image_override = None
        self._cloud_space_instance_gpu_image_override = None
        self._name = None
        self.discriminator = None
        if can_download_source_code is not None:
            self.can_download_source_code = can_download_source_code
        if cloud_space_instance_cpu_image_override is not None:
            self.cloud_space_instance_cpu_image_override = cloud_space_instance_cpu_image_override
        if cloud_space_instance_gpu_image_override is not None:
            self.cloud_space_instance_gpu_image_override = cloud_space_instance_gpu_image_override
        if name is not None:
            self.name = name

    @property
    def can_download_source_code(self) -> 'bool':
        """Gets the can_download_source_code of this ProjectIdAppsv2Body.  # noqa: E501


        :return: The can_download_source_code of this ProjectIdAppsv2Body.  # noqa: E501
        :rtype: bool
        """
        return self._can_download_source_code

    @can_download_source_code.setter
    def can_download_source_code(self, can_download_source_code: 'bool'):
        """Sets the can_download_source_code of this ProjectIdAppsv2Body.


        :param can_download_source_code: The can_download_source_code of this ProjectIdAppsv2Body.  # noqa: E501
        :type: bool
        """

        self._can_download_source_code = can_download_source_code

    @property
    def cloud_space_instance_cpu_image_override(self) -> 'str':
        """Gets the cloud_space_instance_cpu_image_override of this ProjectIdAppsv2Body.  # noqa: E501


        :return: The cloud_space_instance_cpu_image_override of this ProjectIdAppsv2Body.  # noqa: E501
        :rtype: str
        """
        return self._cloud_space_instance_cpu_image_override

    @cloud_space_instance_cpu_image_override.setter
    def cloud_space_instance_cpu_image_override(
            self, cloud_space_instance_cpu_image_override: 'str'):
        """Sets the cloud_space_instance_cpu_image_override of this ProjectIdAppsv2Body.


        :param cloud_space_instance_cpu_image_override: The cloud_space_instance_cpu_image_override of this ProjectIdAppsv2Body.  # noqa: E501
        :type: str
        """

        self._cloud_space_instance_cpu_image_override = cloud_space_instance_cpu_image_override

    @property
    def cloud_space_instance_gpu_image_override(self) -> 'str':
        """Gets the cloud_space_instance_gpu_image_override of this ProjectIdAppsv2Body.  # noqa: E501


        :return: The cloud_space_instance_gpu_image_override of this ProjectIdAppsv2Body.  # noqa: E501
        :rtype: str
        """
        return self._cloud_space_instance_gpu_image_override

    @cloud_space_instance_gpu_image_override.setter
    def cloud_space_instance_gpu_image_override(
            self, cloud_space_instance_gpu_image_override: 'str'):
        """Sets the cloud_space_instance_gpu_image_override of this ProjectIdAppsv2Body.


        :param cloud_space_instance_gpu_image_override: The cloud_space_instance_gpu_image_override of this ProjectIdAppsv2Body.  # noqa: E501
        :type: str
        """

        self._cloud_space_instance_gpu_image_override = cloud_space_instance_gpu_image_override

    @property
    def name(self) -> 'str':
        """Gets the name of this ProjectIdAppsv2Body.  # noqa: E501


        :return: The name of this ProjectIdAppsv2Body.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this ProjectIdAppsv2Body.


        :param name: The name of this ProjectIdAppsv2Body.  # noqa: E501
        :type: str
        """

        self._name = name

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
        if issubclass(ProjectIdAppsv2Body, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'ProjectIdAppsv2Body') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, ProjectIdAppsv2Body):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProjectIdAppsv2Body') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
