import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from aitoolbox import TrainLoopCheckpointEndSave, TTModel
from aitoolbox.torchtrain.callbacks.performance_eval import (ModelPerformanceEvaluation, ModelPerformancePrintReport,
                                                             ModelTrainHistoryPlot, ModelTrainHistoryFileWriter)
from aitoolbox.torchtrain.callbacks.basic import EarlyStopping
from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Net(TTModel):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_loss(self, batch_data, criterion, device):
        data, target = batch_data
        data, target = data.to(device), target.to(device)

        output = self(data)
        loss = criterion(output, target)

        return loss

    def get_predictions(self, batch_data, device):
        data, y_test = batch_data
        data = data.to(device)

        output = self(data)
        y_pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability

        return y_pred.cpu(), y_test, {}


if __name__ == '__main__':
    train_loader = DataLoader(
        datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ),
        batch_size=100)

    val_loader = DataLoader(
        datasets.MNIST(
            './data', train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ),
        batch_size=100)

    test_loader = DataLoader(
        datasets.MNIST(
            './data', train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ),
        batch_size=100)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    callbacks = [ModelPerformanceEvaluation(ClassificationResultPackage(), {},
                                            on_train_data=True, on_val_data=True),
                 ModelPerformancePrintReport(['train_Accuracy', 'val_Accuracy']),
                 ModelTrainHistoryPlot(),
                 ModelTrainHistoryFileWriter(),
                 EarlyStopping(patience=5)]

    print('Starting train loop')
    # With AWS S3 cloud results saving
    TrainLoopCheckpointEndSave(
        model,
        train_loader, val_loader, test_loader,
        optimizer, criterion,
        project_name='tl_full_experiment_tracking', experiment_name='tutorial_example',
        local_model_result_folder_path=THIS_DIR,
        hyperparams={},
        val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
        cloud_save_mode='s3', bucket_name='results',  # bucket_name should be set to the bucket on your S3
        rm_subopt_local_models=True, num_best_checkpoints_kept=3
    ).fit(num_epochs=10, callbacks=callbacks)

    # With only local results saving - no saving to cloud
    # TrainLoopCheckpointEndSave(
    #     model,
    #     train_loader, val_loader, test_loader,
    #     optimizer, criterion,
    #     project_name='tl_full_experiment_tracking', experiment_name='tutorial_example',
    #     local_model_result_folder_path=THIS_DIR,
    #     hyperparams={},
    #     val_result_package=ClassificationResultPackage(), test_result_package=ClassificationResultPackage(),
    #     cloud_save_mode=None,
    #     rm_subopt_local_models=True, num_best_checkpoints_kept=3
    # ).fit(num_epochs=10, callbacks=callbacks)
