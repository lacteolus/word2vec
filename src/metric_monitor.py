from collections import defaultdict
import numpy as np


class MetricMonitor:
    """
    Monitors metrics through the batch
    """

    def __init__(self, epochs: int = 1, float_precision: int = 3):
        self.current_epoch = 1
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
        :param metric_name: Name of the metric
        :param value: Metric value
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
        supported_steps = ["val", "train"]
        if step in supported_steps:
            self.current_step = step
        else:
            raise ValueError(f"Unexpected step name: {step}. Supported steps: {supported_steps}")

    def __str__(self):
        epoch_info = f"Epoch: {self.current_epoch}/{len(self.metrics.keys())} - {self.current_step.capitalize()}."
        metric_info = " | ".join(
                    [
                        f"{metric_name}: {np.mean(metric):.{self.float_precision}f}"
                        for (metric_name, metric) in self.metrics[self.current_epoch][self.current_step].items()
                    ]
                )
        return f"{epoch_info} {metric_info}"
