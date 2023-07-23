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


class V1DownloadModelResponse(object):
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
        'download_url': 'str',
        'metadata': 'dict(str, str)',
        'version': 'str'
    }

    attribute_map = {
        'download_url': 'downloadUrl',
        'metadata': 'metadata',
        'version': 'version'
    }

    def __init__(self,
                 download_url: 'str' = None,
                 metadata: 'dict(str, str)' = None,
                 version: 'str' = None):  # noqa: E501
        """V1DownloadModelResponse - a model defined in Swagger"""  # noqa: E501
        self._download_url = None
        self._metadata = None
        self._version = None
        self.discriminator = None
        if download_url is not None:
            self.download_url = download_url
        if metadata is not None:
            self.metadata = metadata
        if version is not None:
            self.version = version

    @property
    def download_url(self) -> 'str':
        """Gets the download_url of this V1DownloadModelResponse.  # noqa: E501


        :return: The download_url of this V1DownloadModelResponse.  # noqa: E501
        :rtype: str
        """
        return self._download_url

    @download_url.setter
    def download_url(self, download_url: 'str'):
        """Sets the download_url of this V1DownloadModelResponse.


        :param download_url: The download_url of this V1DownloadModelResponse.  # noqa: E501
        :type: str
        """

        self._download_url = download_url

    @property
    def metadata(self) -> 'dict(str, str)':
        """Gets the metadata of this V1DownloadModelResponse.  # noqa: E501


        :return: The metadata of this V1DownloadModelResponse.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: 'dict(str, str)'):
        """Sets the metadata of this V1DownloadModelResponse.


        :param metadata: The metadata of this V1DownloadModelResponse.  # noqa: E501
        :type: dict(str, str)
        """

        self._metadata = metadata

    @property
    def version(self) -> 'str':
        """Gets the version of this V1DownloadModelResponse.  # noqa: E501


        :return: The version of this V1DownloadModelResponse.  # noqa: E501
        :rtype: str
        """
        return self._version

    @version.setter
    def version(self, version: 'str'):
        """Sets the version of this V1DownloadModelResponse.


        :param version: The version of this V1DownloadModelResponse.  # noqa: E501
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
        if issubclass(V1DownloadModelResponse, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1DownloadModelResponse') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1DownloadModelResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1DownloadModelResponse') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
