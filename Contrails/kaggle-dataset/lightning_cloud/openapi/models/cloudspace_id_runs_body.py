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


class CloudspaceIdRunsBody(object):
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
        'app_entrypoint_file': 'str',
        'cluster_id': 'str',
        'dependency_cache_key': 'str',
        'description': 'str',
        'enable_app_server': 'bool',
        'env': 'list[V1EnvVar]',
        'flow_servers': 'list[V1Flowserver]',
        'image_id': 'str',
        'image_spec': 'V1ImageSpec',
        'is_headless': 'bool',
        'is_published': 'bool',
        'local_source': 'bool',
        'network_config': 'list[V1NetworkConfig]',
        'parent_id': 'str',
        'should_mount_cloudspace_content': 'bool',
        'source_code_url': 'str',
        'user_id': 'str',
        'user_requested_flow_compute_config':
        'V1UserRequestedFlowComputeConfig',
        'works': 'list[V1Work]'
    }

    attribute_map = {
        'app_entrypoint_file': 'appEntrypointFile',
        'cluster_id': 'clusterId',
        'dependency_cache_key': 'dependencyCacheKey',
        'description': 'description',
        'enable_app_server': 'enableAppServer',
        'env': 'env',
        'flow_servers': 'flowServers',
        'image_id': 'imageId',
        'image_spec': 'imageSpec',
        'is_headless': 'isHeadless',
        'is_published': 'isPublished',
        'local_source': 'localSource',
        'network_config': 'networkConfig',
        'parent_id': 'parentId',
        'should_mount_cloudspace_content': 'shouldMountCloudspaceContent',
        'source_code_url': 'sourceCodeUrl',
        'user_id': 'userId',
        'user_requested_flow_compute_config': 'userRequestedFlowComputeConfig',
        'works': 'works'
    }

    def __init__(self,
                 app_entrypoint_file: 'str' = None,
                 cluster_id: 'str' = None,
                 dependency_cache_key: 'str' = None,
                 description: 'str' = None,
                 enable_app_server: 'bool' = None,
                 env: 'list[V1EnvVar]' = None,
                 flow_servers: 'list[V1Flowserver]' = None,
                 image_id: 'str' = None,
                 image_spec: 'V1ImageSpec' = None,
                 is_headless: 'bool' = None,
                 is_published: 'bool' = None,
                 local_source: 'bool' = None,
                 network_config: 'list[V1NetworkConfig]' = None,
                 parent_id: 'str' = None,
                 should_mount_cloudspace_content: 'bool' = None,
                 source_code_url: 'str' = None,
                 user_id: 'str' = None,
                 user_requested_flow_compute_config:
                 'V1UserRequestedFlowComputeConfig' = None,
                 works: 'list[V1Work]' = None):  # noqa: E501
        """CloudspaceIdRunsBody - a model defined in Swagger"""  # noqa: E501
        self._app_entrypoint_file = None
        self._cluster_id = None
        self._dependency_cache_key = None
        self._description = None
        self._enable_app_server = None
        self._env = None
        self._flow_servers = None
        self._image_id = None
        self._image_spec = None
        self._is_headless = None
        self._is_published = None
        self._local_source = None
        self._network_config = None
        self._parent_id = None
        self._should_mount_cloudspace_content = None
        self._source_code_url = None
        self._user_id = None
        self._user_requested_flow_compute_config = None
        self._works = None
        self.discriminator = None
        if app_entrypoint_file is not None:
            self.app_entrypoint_file = app_entrypoint_file
        if cluster_id is not None:
            self.cluster_id = cluster_id
        if dependency_cache_key is not None:
            self.dependency_cache_key = dependency_cache_key
        if description is not None:
            self.description = description
        if enable_app_server is not None:
            self.enable_app_server = enable_app_server
        if env is not None:
            self.env = env
        if flow_servers is not None:
            self.flow_servers = flow_servers
        if image_id is not None:
            self.image_id = image_id
        if image_spec is not None:
            self.image_spec = image_spec
        if is_headless is not None:
            self.is_headless = is_headless
        if is_published is not None:
            self.is_published = is_published
        if local_source is not None:
            self.local_source = local_source
        if network_config is not None:
            self.network_config = network_config
        if parent_id is not None:
            self.parent_id = parent_id
        if should_mount_cloudspace_content is not None:
            self.should_mount_cloudspace_content = should_mount_cloudspace_content
        if source_code_url is not None:
            self.source_code_url = source_code_url
        if user_id is not None:
            self.user_id = user_id
        if user_requested_flow_compute_config is not None:
            self.user_requested_flow_compute_config = user_requested_flow_compute_config
        if works is not None:
            self.works = works

    @property
    def app_entrypoint_file(self) -> 'str':
        """Gets the app_entrypoint_file of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The app_entrypoint_file of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._app_entrypoint_file

    @app_entrypoint_file.setter
    def app_entrypoint_file(self, app_entrypoint_file: 'str'):
        """Sets the app_entrypoint_file of this CloudspaceIdRunsBody.


        :param app_entrypoint_file: The app_entrypoint_file of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._app_entrypoint_file = app_entrypoint_file

    @property
    def cluster_id(self) -> 'str':
        """Gets the cluster_id of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The cluster_id of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._cluster_id

    @cluster_id.setter
    def cluster_id(self, cluster_id: 'str'):
        """Sets the cluster_id of this CloudspaceIdRunsBody.


        :param cluster_id: The cluster_id of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._cluster_id = cluster_id

    @property
    def dependency_cache_key(self) -> 'str':
        """Gets the dependency_cache_key of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The dependency_cache_key of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._dependency_cache_key

    @dependency_cache_key.setter
    def dependency_cache_key(self, dependency_cache_key: 'str'):
        """Sets the dependency_cache_key of this CloudspaceIdRunsBody.


        :param dependency_cache_key: The dependency_cache_key of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._dependency_cache_key = dependency_cache_key

    @property
    def description(self) -> 'str':
        """Gets the description of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The description of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: 'str'):
        """Sets the description of this CloudspaceIdRunsBody.


        :param description: The description of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def enable_app_server(self) -> 'bool':
        """Gets the enable_app_server of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The enable_app_server of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: bool
        """
        return self._enable_app_server

    @enable_app_server.setter
    def enable_app_server(self, enable_app_server: 'bool'):
        """Sets the enable_app_server of this CloudspaceIdRunsBody.


        :param enable_app_server: The enable_app_server of this CloudspaceIdRunsBody.  # noqa: E501
        :type: bool
        """

        self._enable_app_server = enable_app_server

    @property
    def env(self) -> 'list[V1EnvVar]':
        """Gets the env of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The env of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: list[V1EnvVar]
        """
        return self._env

    @env.setter
    def env(self, env: 'list[V1EnvVar]'):
        """Sets the env of this CloudspaceIdRunsBody.


        :param env: The env of this CloudspaceIdRunsBody.  # noqa: E501
        :type: list[V1EnvVar]
        """

        self._env = env

    @property
    def flow_servers(self) -> 'list[V1Flowserver]':
        """Gets the flow_servers of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The flow_servers of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: list[V1Flowserver]
        """
        return self._flow_servers

    @flow_servers.setter
    def flow_servers(self, flow_servers: 'list[V1Flowserver]'):
        """Sets the flow_servers of this CloudspaceIdRunsBody.


        :param flow_servers: The flow_servers of this CloudspaceIdRunsBody.  # noqa: E501
        :type: list[V1Flowserver]
        """

        self._flow_servers = flow_servers

    @property
    def image_id(self) -> 'str':
        """Gets the image_id of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The image_id of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._image_id

    @image_id.setter
    def image_id(self, image_id: 'str'):
        """Sets the image_id of this CloudspaceIdRunsBody.


        :param image_id: The image_id of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._image_id = image_id

    @property
    def image_spec(self) -> 'V1ImageSpec':
        """Gets the image_spec of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The image_spec of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: V1ImageSpec
        """
        return self._image_spec

    @image_spec.setter
    def image_spec(self, image_spec: 'V1ImageSpec'):
        """Sets the image_spec of this CloudspaceIdRunsBody.


        :param image_spec: The image_spec of this CloudspaceIdRunsBody.  # noqa: E501
        :type: V1ImageSpec
        """

        self._image_spec = image_spec

    @property
    def is_headless(self) -> 'bool':
        """Gets the is_headless of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The is_headless of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: bool
        """
        return self._is_headless

    @is_headless.setter
    def is_headless(self, is_headless: 'bool'):
        """Sets the is_headless of this CloudspaceIdRunsBody.


        :param is_headless: The is_headless of this CloudspaceIdRunsBody.  # noqa: E501
        :type: bool
        """

        self._is_headless = is_headless

    @property
    def is_published(self) -> 'bool':
        """Gets the is_published of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The is_published of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: bool
        """
        return self._is_published

    @is_published.setter
    def is_published(self, is_published: 'bool'):
        """Sets the is_published of this CloudspaceIdRunsBody.


        :param is_published: The is_published of this CloudspaceIdRunsBody.  # noqa: E501
        :type: bool
        """

        self._is_published = is_published

    @property
    def local_source(self) -> 'bool':
        """Gets the local_source of this CloudspaceIdRunsBody.  # noqa: E501

        Indicates that the client wishes to upload the directory that has the source code. In this case, app's status is populated with the presigned URL and the app's spec is set by the server to the S3 tarball that holds the sourcecode.  # noqa: E501

        :return: The local_source of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: bool
        """
        return self._local_source

    @local_source.setter
    def local_source(self, local_source: 'bool'):
        """Sets the local_source of this CloudspaceIdRunsBody.

        Indicates that the client wishes to upload the directory that has the source code. In this case, app's status is populated with the presigned URL and the app's spec is set by the server to the S3 tarball that holds the sourcecode.  # noqa: E501

        :param local_source: The local_source of this CloudspaceIdRunsBody.  # noqa: E501
        :type: bool
        """

        self._local_source = local_source

    @property
    def network_config(self) -> 'list[V1NetworkConfig]':
        """Gets the network_config of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The network_config of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: list[V1NetworkConfig]
        """
        return self._network_config

    @network_config.setter
    def network_config(self, network_config: 'list[V1NetworkConfig]'):
        """Sets the network_config of this CloudspaceIdRunsBody.


        :param network_config: The network_config of this CloudspaceIdRunsBody.  # noqa: E501
        :type: list[V1NetworkConfig]
        """

        self._network_config = network_config

    @property
    def parent_id(self) -> 'str':
        """Gets the parent_id of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The parent_id of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._parent_id

    @parent_id.setter
    def parent_id(self, parent_id: 'str'):
        """Sets the parent_id of this CloudspaceIdRunsBody.


        :param parent_id: The parent_id of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._parent_id = parent_id

    @property
    def should_mount_cloudspace_content(self) -> 'bool':
        """Gets the should_mount_cloudspace_content of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The should_mount_cloudspace_content of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: bool
        """
        return self._should_mount_cloudspace_content

    @should_mount_cloudspace_content.setter
    def should_mount_cloudspace_content(
            self, should_mount_cloudspace_content: 'bool'):
        """Sets the should_mount_cloudspace_content of this CloudspaceIdRunsBody.


        :param should_mount_cloudspace_content: The should_mount_cloudspace_content of this CloudspaceIdRunsBody.  # noqa: E501
        :type: bool
        """

        self._should_mount_cloudspace_content = should_mount_cloudspace_content

    @property
    def source_code_url(self) -> 'str':
        """Gets the source_code_url of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The source_code_url of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._source_code_url

    @source_code_url.setter
    def source_code_url(self, source_code_url: 'str'):
        """Sets the source_code_url of this CloudspaceIdRunsBody.


        :param source_code_url: The source_code_url of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._source_code_url = source_code_url

    @property
    def user_id(self) -> 'str':
        """Gets the user_id of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The user_id of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: str
        """
        return self._user_id

    @user_id.setter
    def user_id(self, user_id: 'str'):
        """Sets the user_id of this CloudspaceIdRunsBody.


        :param user_id: The user_id of this CloudspaceIdRunsBody.  # noqa: E501
        :type: str
        """

        self._user_id = user_id

    @property
    def user_requested_flow_compute_config(
            self) -> 'V1UserRequestedFlowComputeConfig':
        """Gets the user_requested_flow_compute_config of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The user_requested_flow_compute_config of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: V1UserRequestedFlowComputeConfig
        """
        return self._user_requested_flow_compute_config

    @user_requested_flow_compute_config.setter
    def user_requested_flow_compute_config(
        self,
        user_requested_flow_compute_config: 'V1UserRequestedFlowComputeConfig'
    ):
        """Sets the user_requested_flow_compute_config of this CloudspaceIdRunsBody.


        :param user_requested_flow_compute_config: The user_requested_flow_compute_config of this CloudspaceIdRunsBody.  # noqa: E501
        :type: V1UserRequestedFlowComputeConfig
        """

        self._user_requested_flow_compute_config = user_requested_flow_compute_config

    @property
    def works(self) -> 'list[V1Work]':
        """Gets the works of this CloudspaceIdRunsBody.  # noqa: E501


        :return: The works of this CloudspaceIdRunsBody.  # noqa: E501
        :rtype: list[V1Work]
        """
        return self._works

    @works.setter
    def works(self, works: 'list[V1Work]'):
        """Sets the works of this CloudspaceIdRunsBody.


        :param works: The works of this CloudspaceIdRunsBody.  # noqa: E501
        :type: list[V1Work]
        """

        self._works = works

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
        if issubclass(CloudspaceIdRunsBody, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other: 'CloudspaceIdRunsBody') -> bool:
        """Returns true if both objects are equal"""
        if not isinstance(other, CloudspaceIdRunsBody):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CloudspaceIdRunsBody') -> bool:
        """Returns true if both objects are not equal"""
        return not self == other