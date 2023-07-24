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


class V1ModelVersionArchive(object):
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
        'created_at': 'datetime',
        'downloads': 'str',
        'metadata': 'dict(str, str)',
        'model_id': 'str',
        'project_id': 'str',
        'updated_at': 'datetime',
        'version': 'str'
    }

    attribute_map = {
        'cluster_id': 'clusterId',
        'created_at': 'createdAt',
        'downloads': 'downloads',
        'metadata': 'metadata',
        'model_id': 'modelId',
        'project_id': 'projectId',
        'updated_at': 'updatedAt',
        'version': 'version'
    }

    def __init__(self,
                 cluster_id: 'str' = None,
                 created_at: 'datetime' = None,
                 downloads: 'str' = None,
                 metadata: 'dict(str, str)' = None,
                 model_id: 'str' = None,
                 project_id: 'str' = None,
                 updated_at: 'datetime' = None,
                 version: 'str' = None):  # noqa: E501
        """V1ModelVersionArchive - a model defined in Swagger"""  # noqa: E501
        self._cluster_id = None
        self._created_at = None
        self._downloads = None
        self._metadata = None
        self._model_id = None
        self._project_id = None
        self._updated_at = None
        self._version = None
        self.discriminator = None
        if cluster_id is not None:
            self.cluster_id = cluster_id
        if created_at is not None:
            self.created_at = created_at
        if downloads is not None:
            self.downloads = downloads
        if metadata is not None:
            self.metadata = metadata
        if model_id is not None:
            self.model_id = model_id
        if project_id is not None:
            self.project_id = project_id
        if updated_at is not None:
            self.updated_at = updated_at
        if version is not None:
            self.version = version

    @property
    def cluster_id(self) -> 'str':
        """Gets the cluster_id of this V1ModelVersionArchive.  # noqa: E501


        :return: The cluster_id of this V1ModelVersionArchive.  # noqa: E501
        :rtype: str
        """
        return self._cluster_id

    @cluster_id.setter
    def cluster_id(self, cluster_id: 'str'):
        """Sets the cluster_id of this V1ModelVersionArchive.


        :param cluster_id: The cluster_id of this V1ModelVersionArchive.  # noqa: E501
        :type: str
        """

        self._cluster_id = cluster_id

    @property
    def created_at(self) -> 'datetime':
        """Gets the created_at of this V1ModelVersionArchive.  # noqa: E501


        :return: The created_at of this V1ModelVersionArchive.  # noqa: E501
        :rtype: datetime
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at: 'datetime'):
        """Sets the created_at of this V1ModelVersionArchive.


        :param created_at: The created_at of this V1ModelVersionArchive.  # noqa: E501
        :type: datetime
        """

        self._created_at = created_at

    @property
    def downloads(self) -> 'str':
        """Gets the downloads of this V1ModelVersionArchive.  # noqa: E501


        :return: The downloads of this V1ModelVersionArchive.  # noqa: E501
        :rtype: str
        """
        return self._downloads

    @downloads.setter
    def downloads(self, downloads: 'str'):
        """Sets the downloads of this V1ModelVersionArchive.


        :param downloads: The downloads of this V1ModelVersionArchive.  # noqa: E501
        :type: str
        """

        self._downloads = downloads

    @property
    def metadata(self) -> 'dict(str, str)':
        """Gets the metadata of this V1ModelVersionArchive.  # noqa: E501


        :return: The metadata of this V1ModelVersionArchive.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: 'dict(str, str)'):
        """Sets the metadata of this V1ModelVersionArchive.


        :param metadata: The metadata of this V1ModelVersionArchive.  # noqa: E501
        :type: dict(str, str)
        """

        self._metadata = metadata

    @property
    def model_id(self) -> 'str':
        """Gets the model_id of this V1ModelVersionArchive.  # noqa: E501


        :return: The model_id of this V1ModelVersionArchive.  # noqa: E501
        :rtype: str
        """
        return self._model_id

    @model_id.setter
    def model_id(self, model_id: 'str'):
        """Sets the model_id of this V1ModelVersionArchive.


        :param model_id: The model_id of this V1ModelVersionArchive.  # noqa: E501
        :type: str
        """

        self._model_id = model_id

    @property
    def project_id(self) -> 'str':
        """Gets the project_id of this V1ModelVersionArchive.  # noqa: E501


        :return: The project_id of this V1ModelVersionArchive.  # noqa: E501
        :rtype: str
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id: 'str'):
        """Sets the project_id of this V1ModelVersionArchive.


        :param project_id: The project_id of this V1ModelVersionArchive.  # noqa: E501
        :type: str
        """

        self._project_id = project_id

    @property
    def updated_at(self) -> 'datetime':
        """Gets the updated_at of this V1ModelVersionArchive.  # noqa: E501


        :return: The updated_at of this V1ModelVersionArchive.  # noqa: E501
        :rtype: datetime
        """
        return self._updated_at

    @updated_at.setter
    def updated_at(self, updated_at: 'datetime'):
        """Sets the updated_at of this V1ModelVersionArchive.


        :param updated_at: The updated_at of this V1ModelVersionArchive.  # noqa: E501
        :type: datetime
        """

        self._updated_at = updated_at

    @property
    def version(self) -> 'str':
        """Gets the version of this V1ModelVersionArchive.  # noqa: E501


        :return: The version of this V1ModelVersionArchive.  # noqa: E501
        :rtype: str
        """
        return self._version

    @version.setter
    def version(self, version: 'str'):
        """Sets the version of this V1ModelVersionArchive.


        :param version: The version of this V1ModelVersionArchive.  # noqa: E501
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
        if issubclass(V1ModelVersionArchive, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'V1ModelVersionArchive') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, V1ModelVersionArchive):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'V1ModelVersionArchive') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other