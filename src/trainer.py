import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.metric_monitor import MetricMonitor
from tqdm.auto import tqdm


class Trainer:
    """
    Class for model training and validation
    """

    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            criterion,
            optimizer,
            device,
            metric_monitor: MetricMonitor,
            epochs: int,
    ):
        """
        :param model: Model to be trained
        :param epochs: Number of epochs
        :param train_dataloader: Train dataloader
        :param val_dataloader: Validation dataloader
        :param criterion: Criterion (loss) function
        :param optimizer: Optimizer function
        :param device: Device on which training should be performed
        :param metric_monitor: MetricMonitor to track metrics
        """
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metric_monitor = metric_monitor

        self.model.to(self.device)

    def train(self) -> None:
        """
        Perform training and validation through epochs
        :return: None
        """
        for epoch in range(1, self.epochs + 1):
            self.metric_monitor.set_current_epoch(epoch)
            self.metric_monitor.set_current_step("train")
            self._train_epoch()
            self.metric_monitor.set_current_step("val")
            self._validate_epoch()

    def _train_epoch(self) -> None:
        """
        Perform training for a single epoch
        :return: None
        """
        self.model.train()

        stream = tqdm(self.train_dataloader)

        for i, batch_data in enumerate(stream, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.metric_monitor.update("Loss", loss.item())

            # Description in tqdm progress bar
            stream.set_description(f"{self.metric_monitor}")

    def _validate_epoch(self) -> None:
        """
        Perform validation for a single epoch
        :return: None
        """
        self.model.eval()

        stream = tqdm(self.val_dataloader)

        with torch.no_grad():
            for i, batch_data in enumerate(stream, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.metric_monitor.update("Loss", loss.item())

                stream.set_description(f"{self.metric_monitor}")
