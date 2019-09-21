from abc import ABC, abstractmethod
import boto3
import botocore
import os
import shutil
import zipfile
import tarfile


class BaseDataSaver:
    def __init__(self, bucket_name='model-result'):
        """Base class implementing S3 file saving logic

        Args:
            bucket_name (str): S3 bucket into which the files will be saved
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

    def save_file(self, local_file_path, cloud_file_path):
        """Save file on local drive to the AWS S3

        Args:
            local_file_path (str): path to the file on the local drive
            cloud_file_path (str): destination where the file will be saved on S3 inside the specified bucket

        Returns:
            None
        """
        self.s3_client.upload_file(os.path.expanduser(local_file_path),
                                   self.bucket_name, cloud_file_path)


class BaseDataLoader:
    def __init__(self, bucket_name='dataset-store', local_base_data_folder_path='~/project/data'):
        """Base class implementing S3 file downloading logic

        Args:
            bucket_name (str): S3 bucket from which the files will be downloaded
            local_base_data_folder_path (str): local main experiment saving folder
        """
        self.bucket_name = bucket_name
        self.s3 = boto3.resource('s3')

        self.local_base_data_folder_path = os.path.expanduser(local_base_data_folder_path)
        self.available_prepocessed_datasets = []

    def load_file(self, cloud_file_path, local_file_path):
        """Download the file AWS S3 to the local drive

        Args:
            cloud_file_path (str): location where the file is saved on S3 inside the specified bucket
            local_file_path (str): destination path where the file will be downloaded to the local drive

        Returns:
            None
        """
        local_file_path = os.path.expanduser(local_file_path)

        if os.path.isfile(local_file_path):
            print('File already exists on local disk. Not downloading from S3')
        else:
            print('Local file does not exist on the local disk. Downloading from S3')
            try:
                self.s3.Bucket(self.bucket_name).download_file(cloud_file_path, local_file_path)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist on S3.")
                else:
                    raise

    def exists_local_data_folder(self, data_folder_name, protect_local_folder=True):
        """Check if a specific folder exists in the base data folder

        For example, Squad dataset folder inside /data folder, or pretrained_models folder inside /model_results folder

        Args:
            data_folder_name (str):
            protect_local_folder (bool):

        Returns:
            bool:
        """
        if os.path.exists(os.path.join(self.local_base_data_folder_path, data_folder_name)) and protect_local_folder:
            print('Local folder exists and protect_local_folder in True: leaving local folder as it is.')
            return True

        if os.path.exists(os.path.join(self.local_base_data_folder_path, data_folder_name)) and not protect_local_folder:
            print('Local folder exists and protect_local_folder in False: deleting local folder copy')
            shutil.rmtree(os.path.join(self.local_base_data_folder_path, data_folder_name))

        os.mkdir(os.path.join(self.local_base_data_folder_path, data_folder_name))
        return False

    def preproc_dataset_available(self, preproc_dataset_name):
        return preproc_dataset_name in self.available_prepocessed_datasets

    @staticmethod
    def unzip_file(file_path, target_dir_path):
        """Util function for zip file unzipping

        Args:
            file_path (str): path to the zip file
            target_dir_path (str): destination where unzipped content is stored
        """
        if file_path[-4:] == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir_path)
        elif file_path[-7:] == '.tar.gz':
            with tarfile.open(file_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir_path)


class AbstractDatasetFetcher(ABC):
    @abstractmethod
    def fetch_dataset(self, protect_local_folder=True):
        """

        Args:
            protect_local_folder (bool):

        Returns:
            None
        """
        pass


class SQuAD2DatasetFetcher(AbstractDatasetFetcher, BaseDataLoader):
    def __init__(self, bucket_name='dataset-store', local_dataset_folder_path='~/project/data'):
        """

        Args:
            bucket_name (str):
            local_dataset_folder_path (str):
        """
        BaseDataLoader.__init__(self, bucket_name, local_dataset_folder_path)

    def fetch_dataset(self, protect_local_folder=True):
        """

        Args:
            protect_local_folder (bool):

        Returns:
            None
        """
        if not self.exists_local_data_folder('SQuAD2', protect_local_folder):
            self.load_file(cloud_file_path='SQuAD2/train-v2.0.json',
                           local_file_path=os.path.join(self.local_base_data_folder_path, 'SQuAD2', 'train-v2.0.json'))
            self.load_file(cloud_file_path='SQuAD2/dev-v2.0.json',
                           local_file_path=os.path.join(self.local_base_data_folder_path, 'SQuAD2', 'dev-v2.0.json'))


class QAngarooDatasetFetcher(AbstractDatasetFetcher, BaseDataLoader):
    def __init__(self, bucket_name='dataset-store', local_dataset_folder_path='~/project/data'):
        """

        Args:
            bucket_name (str):
            local_dataset_folder_path (str):
        """
        BaseDataLoader.__init__(self, bucket_name, local_dataset_folder_path)

    def fetch_dataset(self, protect_local_folder=True):
        """

        Args:
            protect_local_folder (bool):

        Returns:
            None
        """
        if not self.exists_local_data_folder('qangaroo_v1', protect_local_folder):
            medhop_local_folder_path = os.path.join(self.local_base_data_folder_path, 'qangaroo_v1')
            wikihop_local_folder_path = os.path.join(self.local_base_data_folder_path, 'qangaroo_v1')
            medhop_local_file_path = os.path.join(medhop_local_folder_path, 'medhop.zip')
            wikihop_local_file_path = os.path.join(wikihop_local_folder_path, 'wikihop.zip')
            self.load_file(cloud_file_path='qangaroo_v1/medhop.zip',
                           local_file_path=medhop_local_file_path)
            self.load_file(cloud_file_path='qangaroo_v1/wikihop.zip',
                           local_file_path=wikihop_local_file_path)
            self.unzip_file(medhop_local_file_path, medhop_local_folder_path)
            self.unzip_file(wikihop_local_file_path, wikihop_local_folder_path)


class CNNDailyMailDatasetFetcher(AbstractDatasetFetcher, BaseDataLoader):
    def __init__(self, bucket_name='dataset-store', local_dataset_folder_path='~/project/data'):
        """

        Args:
            bucket_name (str):
            local_dataset_folder_path (str):
        """
        BaseDataLoader.__init__(self, bucket_name, local_dataset_folder_path)
        self.available_prepocessed_datasets = ['abisee', 'danqi']

    def fetch_dataset(self, protect_local_folder=True):
        """

        Args:
            protect_local_folder (bool):

        Returns:
            None
        """
        raise NotImplementedError('Currently the original CNN/DailyMail dataset is not stored on S3. '
                                  'Rather use one of the preprocessed datasets.')
        pass

    def fetch_preprocessed_dataset(self, preprocess_name, protect_local_folder=True):
        """

        Args:
            preprocess_name (str):
            protect_local_folder (bool):

        Returns:
            None
        """
        if not self.preproc_dataset_available(preprocess_name):
            raise ValueError('Preprocessing name not available. Select: abisee / danqi')

        if preprocess_name == 'abisee':
            if not self.exists_local_data_folder('cnn-dailymail-abisee', protect_local_folder):
                cnn_local_folder_path = os.path.join(self.local_base_data_folder_path, 'cnn-dailymail-abisee')
                dm_local_folder_path = os.path.join(self.local_base_data_folder_path, 'cnn-dailymail-abisee')
                cnn_local_file_path = os.path.join(cnn_local_folder_path, 'cnn_stories_tokenized.zip')
                dm_local_file_path = os.path.join(dm_local_folder_path, 'dm_stories_tokenized.zip')

                self.load_file(cloud_file_path='cnn-dailymail/preproc/abisee/cnn_stories_tokenized.zip',
                               local_file_path=cnn_local_file_path)
                self.load_file(cloud_file_path='cnn-dailymail/preproc/abisee/dm_stories_tokenized.zip',
                               local_file_path=dm_local_file_path)
                self.unzip_file(cnn_local_file_path, cnn_local_folder_path)
                self.unzip_file(dm_local_file_path, dm_local_folder_path)

        elif preprocess_name == 'danqi':
            if not self.exists_local_data_folder('cnn-dailymail-danqi', protect_local_folder):
                cnn_local_folder_path = os.path.join(self.local_base_data_folder_path, 'cnn-dailymail-danqi')
                dm_local_folder_path = os.path.join(self.local_base_data_folder_path, 'cnn-dailymail-danqi')
                cnn_local_file_path = os.path.join(cnn_local_folder_path, 'cnn.tar.gz')
                dm_local_file_path = os.path.join(dm_local_folder_path, 'dailymail.tar.gz')

                self.load_file(cloud_file_path='cnn-dailymail/preproc/danqi/cnn.tar.gz',
                               local_file_path=cnn_local_file_path)
                self.load_file(cloud_file_path='cnn-dailymail/preproc/danqi/dailymail.tar.gz',
                               local_file_path=dm_local_file_path)
                self.unzip_file(cnn_local_file_path, cnn_local_folder_path)
                self.unzip_file(dm_local_file_path, dm_local_folder_path)


class HotpotQADatasetFetcher(AbstractDatasetFetcher, BaseDataLoader):
    def __init__(self, bucket_name='dataset-store', local_dataset_folder_path='~/project/data'):
        """

        https://hotpotqa.github.io/
        https://arxiv.org/pdf/1809.09600.pdf

        https://github.com/hotpotqa/hotpot


        Args:
            bucket_name (str):
            local_dataset_folder_path (str):
        """
        BaseDataLoader.__init__(self, bucket_name, local_dataset_folder_path)

    def fetch_dataset(self, protect_local_folder=True):
        """

        Args:
            protect_local_folder (bool):

        Returns:
            None
        """
        if not self.exists_local_data_folder('HotpotQA', protect_local_folder):
            hotpotqa_local_folder_path = os.path.join(self.local_base_data_folder_path, 'HotpotQA')
            hotpotqa_local_file_path = os.path.join(hotpotqa_local_folder_path, 'HotpotQA.zip')
            self.load_file(cloud_file_path='HotpotQA/HotpotQA.zip',
                           local_file_path=hotpotqa_local_file_path)
            self.unzip_file(hotpotqa_local_file_path, hotpotqa_local_folder_path)
