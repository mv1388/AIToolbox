from AIToolbox.AWS.model_save import PyTorchS3ModelSaver
from AIToolbox.GoogleCloud.model_save import PyTorchGoogleStorageModelSaver
from AIToolbox.experiment_save.local_model_save import PyTorchLocalModelSaver
from AIToolbox.experiment_save.experiment_saver import FullPyTorchExperimentS3Saver, FullPyTorchExperimentGoogleStorageSaver
from AIToolbox.experiment_save.local_experiment_saver import FullPyTorchExperimentLocalSaver
from AIToolbox.experiment_save.training_history import TrainingHistory


class AbstractCallback:
    def __init__(self, callback_name):
        self.callback_name = callback_name
        self.train_loop_obj = None

    def register_train_loop_object(self, train_loop_obj):
        """

        Args:
            train_loop_obj (AIToolbox.torchtrain.train_loop.TrainLoop):

        Returns:

        """
        self.train_loop_obj = train_loop_obj
        self.on_train_loop_registration()
        return self

    def on_train_loop_registration(self):
        """Execute callback initialization / preparation after the train_loop_object becomes available

        Returns:

        """
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass


class EarlyStoppingCallback(AbstractCallback):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0):
        """

        Args:
            monitor (str):
            min_delta (float):
            patience (int):
        """
        AbstractCallback.__init__(self, 'EarlyStopping')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.patience_count = self.patience
        self.best_performance = None
        self.best_epoch = 0

    def on_epoch_end(self):
        """

        Returns:

        """
        history_data = self.train_loop_obj.train_history[self.monitor]
        current_performance = history_data[-1]

        # if len(history_data) > self.patience:
        #     history_window = history_data[-self.patience:]
        #
        #     if 'loss' in self.monitor:
        #         if history_window[0] == min(history_window) and history_window[0] < history_window[-1]-self.patience:
        #             train_loop_obj.early_stop = True
        #     else:
        #         if history_window[0] == max(history_window) and history_window[0] > history_window[-1]+self.patience:
        #             train_loop_obj.early_stop = True

        if self.best_performance is None:
            self.best_performance = current_performance
            self.best_epoch = self.train_loop_obj.epoch

        else:
            if 'loss' in self.monitor:
                if current_performance < self.best_performance - self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = self.train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1
            else:
                if current_performance > self.best_performance + self.min_delta:
                    self.best_performance = current_performance
                    self.best_epoch = self.train_loop_obj.epoch
                    self.patience_count = self.patience
                else:
                    self.patience_count -= 1

            if self.patience_count < 0:
                self.train_loop_obj.early_stop = True

    def on_train_end(self):
        """

        Returns:

        """
        if self.train_loop_obj.early_stop:
            print(f'Early stopping at epoch: {self.train_loop_obj.epoch}. Best recorded epoch: {self.best_epoch}.')


class ModelCheckpointCallback(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path, cloud_save_mode='s3'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            cloud_save_mode (str or None):
        """
        AbstractCallback.__init__(self, 'Model checkpoint at end of epoch')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.cloud_save_mode = cloud_save_mode

        if self.cloud_save_mode == 's3' or self.cloud_save_mode == 'aws_s3' or self.cloud_save_mode == 'aws':
            self.model_checkpointer = PyTorchS3ModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        elif self.cloud_save_mode == 'gcs' or self.cloud_save_mode == 'google_storage' or self.cloud_save_mode == 'google storage':
            self.model_checkpointer = PyTorchGoogleStorageModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path,
                checkpoint_model=True
            )
        else:
            self.model_checkpointer = PyTorchLocalModelSaver(
                local_model_result_folder_path=self.local_model_result_folder_path, checkpoint_model=True
            )

    def on_epoch_end(self):
        """

        Returns:

        """
        self.model_checkpointer.save_model(model=self.train_loop_obj.model,
                                           project_name=self.project_name,
                                           experiment_name=self.experiment_name,
                                           experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                           epoch=self.train_loop_obj.epoch,
                                           protect_existing_folder=True)


class ModelTrainEndSaveCallback(AbstractCallback):
    def __init__(self, project_name, experiment_name, local_model_result_folder_path,
                 args, val_result_package=None, test_result_package=None, cloud_save_mode='s3'):
        """

        Args:
            project_name (str):
            experiment_name (str):
            local_model_result_folder_path (str):
            args (dict):
            val_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            test_result_package (AIToolbox.experiment_save.result_package.abstract_result_packages.AbstractResultPackage):
            cloud_save_mode (str or None):
        """
        AbstractCallback.__init__(self, 'Model save at the end of training')
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_model_result_folder_path = local_model_result_folder_path
        self.args = args
        self.val_result_package = val_result_package
        self.test_result_package = test_result_package
        self.result_package = None
        self.cloud_save_mode = cloud_save_mode

        self.check_result_packages()

        if self.cloud_save_mode == 's3' or self.cloud_save_mode == 'aws_s3' or self.cloud_save_mode == 'aws':
            self.results_saver = FullPyTorchExperimentS3Saver(self.project_name, self.experiment_name,
                                                              local_model_result_folder_path=self.local_model_result_folder_path)

        elif self.cloud_save_mode == 'gcs' or self.cloud_save_mode == 'google_storage' or self.cloud_save_mode == 'google storage':
            self.results_saver = FullPyTorchExperimentGoogleStorageSaver(self.project_name, self.experiment_name,
                                                                         local_model_result_folder_path=self.local_model_result_folder_path)
        else:
            self.results_saver = FullPyTorchExperimentLocalSaver(self.project_name, self.experiment_name,
                                                                 local_model_result_folder_path=self.local_model_result_folder_path)

    def on_train_end(self):
        """

        Returns:

        """
        train_history = self.train_loop_obj.train_history
        epoch_list = list(range(len(self.train_loop_obj.train_history[list(self.train_loop_obj.train_history.keys())[0]])))
        train_hist_pkg = TrainingHistory(train_history, epoch_list)

        if self.val_result_package is not None:
            y_test, y_pred, additional_results = self.train_loop_obj.predict_on_validation_set()
            self.val_result_package.pkg_name += '_VAL'
            self.val_result_package.prepare_result_package(y_test, y_pred,
                                                           hyperparameters=self.args, training_history=train_hist_pkg,
                                                           additional_results=additional_results)
            self.result_package = self.val_result_package

        if self.test_result_package is not None:
            y_test_test, y_pred_test, additional_results_test = self.train_loop_obj.predict_on_test_set()
            self.test_result_package.pkg_name += '_TEST'
            self.test_result_package.prepare_result_package(y_test_test, y_pred_test,
                                                            hyperparameters=self.args, training_history=train_hist_pkg,
                                                            additional_results=additional_results_test)
            self.result_package = self.test_result_package + self.result_package if self.result_package is not None \
                else self.test_result_package

        self.results_saver.save_experiment(self.train_loop_obj.model, self.result_package,
                                           experiment_timestamp=self.train_loop_obj.experiment_timestamp,
                                           save_true_pred_labels=True)

    def on_train_loop_registration(self):
        if self.val_result_package is not None:
            self.val_result_package.set_experiment_dir_path_for_additional_results(self.project_name, self.experiment_name,
                                                                                   self.train_loop_obj.experiment_timestamp,
                                                                                   self.local_model_result_folder_path)
        if self.test_result_package is not None:
            self.test_result_package.set_experiment_dir_path_for_additional_results(self.project_name,
                                                                                    self.experiment_name,
                                                                                    self.train_loop_obj.experiment_timestamp,
                                                                                    self.local_model_result_folder_path)

    def check_result_packages(self):
        if self.val_result_package is None and self.test_result_package is None:
            raise ValueError("Both val_result_package and test_result_package are None. "
                             "At least one of these should be not None but actual result package.")
