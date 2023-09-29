import logging

from torchmetrics.classification.precision_fixed_recall import BinaryPrecisionAtFixedRecall
from torch import Tensor


class PrecisionAtFixedRecall(BinaryPrecisionAtFixedRecall):
    def compute(self) -> Tensor:
        """Computes precision at fixed recall.

        Returns:
            Tensor: Precision at fixed recall.
        """

        precision, threshold = super().compute()
        logging.debug("Precision: %s Threshold:%s", precision, threshold)
        return precision


# this is only necessary because the log connector requires single scalars
class ThresholdAtPrecisionAtFixedRecall(BinaryPrecisionAtFixedRecall):
    def compute(self) -> Tensor:
        """Compuutes the actual recall of the corresponding threshold at fixed recall.

        Returns:
            Tensor: Precision at fixed recall.
        """

        precision, threshold = super().compute()
        logging.debug("Precision: %s Threshold:%s", precision, threshold)
        return threshold
