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


class MultipartCompleteBody(object):
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
        'cluster_id': 'str',
        'filename': 'str',
        'parts': 'list[V1CompleteMultiPartUpload]',
        'upload_id': 'str'
    }

    attribute_map = {
        'cluster_id': 'clusterId',
        'filename': 'filename',
        'parts': 'parts',
        'upload_id': 'uploadId'
    }

    def __init__(self,
                 cluster_id: 'str' = None,
                 filename: 'str' = None,
                 parts: 'list[V1CompleteMultiPartUpload]' = None,
                 upload_id: 'str' = None):  # noqa: E501
        """MultipartCompleteBody - a model defined in Swagger"""  # noqa: E501
        self._cluster_id = None
        self._filename = None
        self._parts = None
        self._upload_id = None
        self.discriminator = None
        if cluster_id is not None:
            self.cluster_id = cluster_id
        if filename is not None:
            self.filename = filename
        if parts is not None:
            self.parts = parts
        if upload_id is not None:
            self.upload_id = upload_id

    @property
    def cluster_id(self) -> 'str':
        """Gets the cluster_id of this MultipartCompleteBody.  # noqa: E501


        :return: The cluster_id of this MultipartCompleteBody.  # noqa: E501
        :rtype: str
        """
        return self._cluster_id

    @cluster_id.setter
    def cluster_id(self, cluster_id: 'str'):
        """Sets the cluster_id of this MultipartCompleteBody.


        :param cluster_id: The cluster_id of this MultipartCompleteBody.  # noqa: E501
        :type: str
        """

        self._cluster_id = cluster_id

    @property
    def filename(self) -> 'str':
        """Gets the filename of this MultipartCompleteBody.  # noqa: E501


        :return: The filename of this MultipartCompleteBody.  # noqa: E501
        :rtype: str
        """
        return self._filename

    @filename.setter
    def filename(self, filename: 'str'):
        """Sets the filename of this MultipartCompleteBody.


        :param filename: The filename of this MultipartCompleteBody.  # noqa: E501
        :type: str
        """

        self._filename = filename

    @property
    def parts(self) -> 'list[V1CompleteMultiPartUpload]':
        """Gets the parts of this MultipartCompleteBody.  # noqa: E501


        :return: The parts of this MultipartCompleteBody.  # noqa: E501
        :rtype: list[V1CompleteMultiPartUpload]
        """
        return self._parts

    @parts.setter
    def parts(self, parts: 'list[V1CompleteMultiPartUpload]'):
        """Sets the parts of this MultipartCompleteBody.


        :param parts: The parts of this MultipartCompleteBody.  # noqa: E501
        :type: list[V1CompleteMultiPartUpload]
        """

        self._parts = parts

    @property
    def upload_id(self) -> 'str':
        """Gets the upload_id of this MultipartCompleteBody.  # noqa: E501


        :return: The upload_id of this MultipartCompleteBody.  # noqa: E501
        :rtype: str
        """
        return self._upload_id

    @upload_id.setter
    def upload_id(self, upload_id: 'str'):
        """Sets the upload_id of this MultipartCompleteBody.


        :param upload_id: The upload_id of this MultipartCompleteBody.  # noqa: E501
        :type: str
        """

        self._upload_id = upload_id

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
        if issubclass(MultipartCompleteBody, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'MultipartCompleteBody') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, MultipartCompleteBody):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'MultipartCompleteBody') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
