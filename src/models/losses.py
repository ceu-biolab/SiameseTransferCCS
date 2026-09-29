"""Losses for scalar Siamese pretraining outputs."""

from keras import ops, saving


@saving.register_keras_serializable(package="SiameseTransferCCS")
def scalar_absolute_error(y_true, y_pred):
    """Return one absolute error per sample for targets/predictions shaped (B,).

    Keras applies sample weights before averaging over the full batch. Unlike
    MAE's last-axis mean, this preserves the sample axis so volume masks exclude
    individual errors and their gradients.
    """
    return ops.abs(y_pred - y_true)
