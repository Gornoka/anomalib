import logging

from torchmetrics.classification.precision_fixed_recall import BinaryPrecisionAtFixedRecall
from torch import Tensor


class PrecisionAtFixedRecall(BinaryPrecisionAtFixedRecall):
    def compute(self) -> Tensor:
        """Computes precision at fixed recall.

        Returns:
            Tensor: Precision at fixed recall.
        """

        precision, recall = super().compute()
        logging.debug("Precision: %s at fixed recall:%s", precision, recall)
        return precision


# this is only necessary because the log connector requires single scalars
class RecallAtPrecisionAtFixedRecall(BinaryPrecisionAtFixedRecall):
    def compute(self) -> Tensor:
        """Compuutes the actual recall of the corresponding precision at fixed recall.

        Returns:
            Tensor: Precision at fixed recall.
        """

        precision, recall = super().compute()
        logging.debug("Precision: %s at fixed recall:%s", precision, recall)
        return recall
