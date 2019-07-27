from abc import ABC, abstractmethod
import os
import time
import datetime

from AIToolbox.cloud.AWS.data_access import BaseDataSaver
from AIToolbox.experiment.local_save.local_model_save import KerasLocalModelSaver, TensorFlowLocalModelSaver, PyTorchLocalModelSaver


class AbstractModelSaver(ABC):
    @abstractmethod
    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        """

        Args:
            model:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str, str): model_s3_path, experiment_timestamp, model_local_path
        """
        pass


class BaseModelSaver(BaseDataSaver):
    def __init__(self, bucket_name='model-result', checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            checkpoint_model (bool):
        """
        BaseDataSaver.__init__(self, bucket_name)
        self.checkpoint_model = checkpoint_model

    def create_experiment_cloud_storage_folder_structure(self, project_name, experiment_name, experiment_timestamp):
        """

        Args:
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str):

        Returns:
            str:
        """
        experiment_cloud_path = os.path.join(project_name,
                                             experiment_name + '_' + experiment_timestamp,
                                             'model' if not self.checkpoint_model else 'checkpoint_model')
        return experiment_cloud_path


class KerasS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelSaver.__init__(self, bucket_name, checkpoint_model)
        self.keras_local_saver = KerasLocalModelSaver(local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        """

        Args:
            model (keras.engine.training.Model):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str, str): model_s3_path, experiment_timestamp, model_local_path

        Examples:
            local_model_result_folder_path = '~/project/model_results'
            # local_model_result_folder_path = '~/PycharmProjects/MemoryNet/model_results'
            m_saver = KerasS3ModelSaver(local_model_result_folder_path=local_model_result_folder_path)
            m_saver.save_model(model=model,
                               project_name='QA_QAngaroo', experiment_name='FastQA_RNN_concat_model_GLOVE',
                               protect_existing_folder=False)
        """
        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        model_name, model_local_path = self.keras_local_saver.save_model(model, project_name, experiment_name,
                                                                         experiment_timestamp, epoch,
                                                                         protect_existing_folder)

        experiment_s3_path = self.create_experiment_cloud_storage_folder_structure(project_name, experiment_name, experiment_timestamp)
        model_s3_path = os.path.join(experiment_s3_path, model_name)

        self.save_file(local_file_path=model_local_path, cloud_file_path=model_s3_path)

        full_model_s3_path = os.path.join(self.bucket_name, model_s3_path)

        return full_model_s3_path, experiment_timestamp, model_local_path


class TensorFlowS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelSaver.__init__(self, bucket_name, checkpoint_model)
        self.tf_local_saver = TensorFlowLocalModelSaver(local_model_result_folder_path, checkpoint_model)

        raise NotImplementedError

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        raise NotImplementedError
        pass


class PyTorchS3ModelSaver(AbstractModelSaver, BaseModelSaver):
    def __init__(self, bucket_name='model-result', local_model_result_folder_path='~/project/model_result',
                 checkpoint_model=False):
        """

        Args:
            bucket_name (str):
            local_model_result_folder_path (str):
            checkpoint_model (bool):
        """
        BaseModelSaver.__init__(self, bucket_name, checkpoint_model)
        self.pytorch_local_saver = PyTorchLocalModelSaver(local_model_result_folder_path, checkpoint_model)

    def save_model(self, model, project_name, experiment_name, experiment_timestamp=None, epoch=None, protect_existing_folder=True):
        """

        Args:
            model (dict):
            project_name (str):
            experiment_name (str):
            experiment_timestamp (str or None):
            epoch (int or None):
            protect_existing_folder (bool):

        Returns:
            (str, str, str): model_s3_path, experiment_timestamp, model_local_path

        Examples:
            local_model_result_folder_path = '~/project/model_results'
            # local_model_result_folder_path = '~/PycharmProjects/MemoryNet/model_results'
            m_saver = PyTorchLocalModelSaver(local_model_result_folder_path=local_model_result_folder_path)
            m_saver.save_model(model=model,
                               project_name='QA_QAngaroo', experiment_name='FastQA_RNN_concat_model_GLOVE',
                               protect_existing_folder=False)
        """
        PyTorchLocalModelSaver.check_model_dict_contents(model)

        if experiment_timestamp is None:
            experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        model_name, model_local_path = self.pytorch_local_saver.save_model(model, project_name, experiment_name,
                                                                           experiment_timestamp, epoch,
                                                                           protect_existing_folder)

        experiment_s3_path = self.create_experiment_cloud_storage_folder_structure(project_name, experiment_name, experiment_timestamp)
        model_s3_path = os.path.join(experiment_s3_path, model_name)

        self.save_file(local_file_path=model_local_path, cloud_file_path=model_s3_path)

        full_model_s3_path = os.path.join(self.bucket_name, model_s3_path)

        return full_model_s3_path, experiment_timestamp, model_local_path
