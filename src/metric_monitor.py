from collections import defaultdict
import numpy as np


class MetricMonitor:
    """
    Monitors metrics through the batch
    """

    def __init__(self, epochs: int, float_precision: int = 3):
        self.current_epoch = 0
        self.current_step = "train"
        self.float_precision = float_precision
        self.metrics = {
            i: {
                "train": defaultdict(lambda: []),
                "val": defaultdict(lambda: [])
            } for i in range(1, epochs + 1)
        }

    def update(self, metric_name: str, value: float):
        """
        Update metric
        """
        metric = self.metrics[self.current_epoch][self.current_step][metric_name]
        metric.append(value)

    def set_current_epoch(self, epoch: int) -> None:
        """
        Set current epoch number
        """
        self.current_epoch = epoch

    def set_current_step(self, step: str) -> None:
        """
        Set current step type: either "val" or "train"
        """
        if step in ["val", "train"]:
            self.current_step = step
        else:
            raise ValueError

    def __str__(self):
        return f"Epoch: {self.current_epoch}. {[self.current_step]} "\
               " | ".join(
                    [
                        f"{metric_name}: {np.mean(metric)}:.self.float_precision"
                        for (metric_name, metric) in self.metrics[self.current_epoch][self.current_step].items()
                    ]
                )
    #
    # @property
    # def results(self):
    #     """
    #     Return current state of metrics as dictionary. Only 'avg' value is used
    #     """
    #     return {metric: values["avg"] for metric, values in self.metrics.items()}
