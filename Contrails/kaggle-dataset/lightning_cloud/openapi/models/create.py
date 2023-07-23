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


class Create(object):
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
        'aws': 'V1AwsDataConnection',
        'cluster_id': 'str',
        'create_index': 'bool',
        'gcp': 'V1GcpDataConnection',
        'id': 'str',
        'name': 'str',
        'run_cmds': 'list[str]'
    }

    attribute_map = {
        'aws': 'aws',
        'cluster_id': 'clusterId',
        'create_index': 'createIndex',
        'gcp': 'gcp',
        'id': 'id',
        'name': 'name',
        'run_cmds': 'runCmds'
    }

    def __init__(self,
                 aws: 'V1AwsDataConnection' = None,
                 cluster_id: 'str' = None,
                 create_index: 'bool' = None,
                 gcp: 'V1GcpDataConnection' = None,
                 id: 'str' = None,
                 name: 'str' = None,
                 run_cmds: 'list[str]' = None):  # noqa: E501
        """Create - a model defined in Swagger"""  # noqa: E501
        self._aws = None
        self._cluster_id = None
        self._create_index = None
        self._gcp = None
        self._id = None
        self._name = None
        self._run_cmds = None
        self.discriminator = None
        if aws is not None:
            self.aws = aws
        if cluster_id is not None:
            self.cluster_id = cluster_id
        if create_index is not None:
            self.create_index = create_index
        if gcp is not None:
            self.gcp = gcp
        if id is not None:
            self.id = id
        if name is not None:
            self.name = name
        if run_cmds is not None:
            self.run_cmds = run_cmds

    @property
    def aws(self) -> 'V1AwsDataConnection':
        """Gets the aws of this Create.  # noqa: E501


        :return: The aws of this Create.  # noqa: E501
        :rtype: V1AwsDataConnection
        """
        return self._aws

    @aws.setter
    def aws(self, aws: 'V1AwsDataConnection'):
        """Sets the aws of this Create.


        :param aws: The aws of this Create.  # noqa: E501
        :type: V1AwsDataConnection
        """

        self._aws = aws

    @property
    def cluster_id(self) -> 'str':
        """Gets the cluster_id of this Create.  # noqa: E501


        :return: The cluster_id of this Create.  # noqa: E501
        :rtype: str
        """
        return self._cluster_id

    @cluster_id.setter
    def cluster_id(self, cluster_id: 'str'):
        """Sets the cluster_id of this Create.


        :param cluster_id: The cluster_id of this Create.  # noqa: E501
        :type: str
        """

        self._cluster_id = cluster_id

    @property
    def create_index(self) -> 'bool':
        """Gets the create_index of this Create.  # noqa: E501


        :return: The create_index of this Create.  # noqa: E501
        :rtype: bool
        """
        return self._create_index

    @create_index.setter
    def create_index(self, create_index: 'bool'):
        """Sets the create_index of this Create.


        :param create_index: The create_index of this Create.  # noqa: E501
        :type: bool
        """

        self._create_index = create_index

    @property
    def gcp(self) -> 'V1GcpDataConnection':
        """Gets the gcp of this Create.  # noqa: E501


        :return: The gcp of this Create.  # noqa: E501
        :rtype: V1GcpDataConnection
        """
        return self._gcp

    @gcp.setter
    def gcp(self, gcp: 'V1GcpDataConnection'):
        """Sets the gcp of this Create.


        :param gcp: The gcp of this Create.  # noqa: E501
        :type: V1GcpDataConnection
        """

        self._gcp = gcp

    @property
    def id(self) -> 'str':
        """Gets the id of this Create.  # noqa: E501


        :return: The id of this Create.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: 'str'):
        """Sets the id of this Create.


        :param id: The id of this Create.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def name(self) -> 'str':
        """Gets the name of this Create.  # noqa: E501


        :return: The name of this Create.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: 'str'):
        """Sets the name of this Create.


        :param name: The name of this Create.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def run_cmds(self) -> 'list[str]':
        """Gets the run_cmds of this Create.  # noqa: E501


        :return: The run_cmds of this Create.  # noqa: E501
        :rtype: list[str]
        """
        return self._run_cmds

    @run_cmds.setter
    def run_cmds(self, run_cmds: 'list[str]'):
        """Sets the run_cmds of this Create.


        :param run_cmds: The run_cmds of this Create.  # noqa: E501
        :type: list[str]
        """

        self._run_cmds = run_cmds

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
        if issubclass(Create, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'Create') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, Create):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Create') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other
