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


class V1UploadModelRequest(object):
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
        'metadata': 'dict(str, str)',
        'name': 'str',
        'private': 'bool',
        'project_id': 'str',
        'version': 'str'
    }

    attribute_map = {
        'metadata': 'metadata',
        'name': 'name',
        'private': 'private',
        'project_id': 'projectId',
        'version': 'version'
    }

    def __init__(self,
                 metadata: 'dict(str, str)' = None,
                 name: 'str' = None,
                 private: 'bool' = None,
                 project_id: 'str' = None,
                 version: 'str' = None):  # noqa: E501
        """V1UploadModelRequest - a model defined in Swagger"""  # noqa: E501
        self._metadata = None
        self._name = None
        self._private = None
        self._project_id = None
        self._version = None
        self.discriminator = None
        if metadata is not None:
            self.metadata = metadata
        if name is not None:
            self.name = name
        if private is not None:
            self.private = private
        if project_id is not None:
            self.project_id = project_id
        if version is not None:
            self.version = version

    @property
    def metadata(self) -> 'dict(str, str)':
        """Gets the metadata of this V1UploadModelRequest.  # noqa: E501


        :return: The metadata of this V1UploadModelRequest.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: 'dict(str, str)'):
        """Sets the metadata of this V1UploadModelRequest.


        :param metadata: The metadata of this V1UploadModelRequest.  # noqa: E501
        :type: dict(str, str)
        """

        self._metadata = metadata

    @property
    def name(self) -> 'str':
        """Gets the name of this V1UploadModelRequest.  # noqa: E501


        :return: The name of this V1UploadModelRequest.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this V1UploadModelRequest.


        :param name: The name of this V1UploadModelRequest.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def private(self) -> 'bool':
        """Gets the private of this V1UploadModelRequest.  # noqa: E501


        :return: The private of this V1UploadModelRequest.  # noqa: E501
        :rtype: bool
        """
        return self._private

    @private.setter
    def private(self, private: 'bool'):
        """Sets the private of this V1UploadModelRequest.


        :param private: The private of this V1UploadModelRequest.  # noqa: E501
        :type: bool
        """

        self._private = private

    @property
    def project_id(self) -> 'str':
        """Gets the project_id of this V1UploadModelRequest.  # noqa: E501


        :return: The project_id of this V1UploadModelRequest.  # noqa: E501
        :rtype: str
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id: 'str'):
        """Sets the project_id of this V1UploadModelRequest.


        :param project_id: The project_id of this V1UploadModelRequest.  # noqa: E501
        :type: str
        """

        self._project_id = project_id

    @property
    def version(self) -> 'str':
        """Gets the version of this V1UploadModelRequest.  # noqa: E501


        :return: The version of this V1UploadModelRequest.  # noqa: E501
        :rtype: str
        """
        return self._version

    @version.setter
    def version(self, version: 'str'):
        """Sets the version of this V1UploadModelRequest.


        :param version: The version of this V1UploadModelRequest.  # noqa: E501
        :type: str
        """

        self._version = version

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
        if issubclass(V1UploadModelRequest, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1UploadModelRequest') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1UploadModelRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1UploadModelRequest') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
