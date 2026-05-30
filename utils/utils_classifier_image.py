import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)




DEFAULT_THRESHOLD = 0.5


def scores_to_binary_predictions(y_scores, threshold=DEFAULT_THRESHOLD):
    return (y_scores >= threshold).astype(int)


def build_cnn_baseline(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape, name="image_input")

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="image_features")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="sigmoid",
        name="multi_label_output",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_image_baseline",
    )


def evaluate_multilabel_predictions(y_true, y_scores, threshold=DEFAULT_THRESHOLD):
    y_pred = scores_to_binary_predictions(y_scores, threshold=threshold)

    metrics = {
        "threshold": threshold,
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "average_precision_micro": average_precision_score(
            y_true,
            y_scores,
            average="micro",
        ),
        "average_precision_macro": average_precision_score(
            y_true,
            y_scores,
            average="macro",
        ),
    }

    try:
        metrics["auc_micro"] = roc_auc_score(y_true, y_scores, average="micro")
        metrics["auc_macro"] = roc_auc_score(y_true, y_scores, average="macro")
    except ValueError:
        metrics["auc_micro"] = np.nan
        metrics["auc_macro"] = np.nan

    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    return metrics, y_pred, confusion_matrices


def metrics_to_dataframe(metrics, model_name):
    return pd.DataFrame([{"model": model_name, **metrics}])


def plot_multilabel_confusion_matrices(
    y_true,
    y_proba,
    class_names,
    threshold=DEFAULT_THRESHOLD,
    ncols=6,
    figsize=None,
):
    y_pred = scores_to_binary_predictions(y_proba, threshold=threshold)
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    ncols = min(ncols, n_classes)
    nrows = int(np.ceil(n_classes / ncols))

    if figsize is None:
        figsize = (3.2 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, (matrix, class_name) in enumerate(zip(confusion_matrices, class_names)):
        tn, fp, fn, tp = matrix.ravel()
        cm_display = np.array([[tn, fp], [fn, tp]])

        sns.heatmap(
            cm_display,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=axes[i],
        )
        axes[i].set_title(f"Label {class_name}")

    for ax in axes[n_classes:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
def build_resnet50_feature_extractor(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape, name="image_input")

    x = tf.keras.layers.Rescaling(255.0, name="rescale_to_255")(inputs)
    x = tf.keras.layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input,
        name="resnet50_preprocessing",
    )(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="image_features")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="sigmoid",
        name="multi_label_output",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="resnet50_feature_extractor",
    )

def unfreeze_last_layers(model, base_model_name="resnet50", trainable_layers=30):
    base_model = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and base_model_name in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError(f"Base model contenant '{base_model_name}' introuvable.")

    base_model.trainable = True

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    for layer in base_model.layers[-trainable_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return model


def build_resnet50_from_scratch(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape, name="image_input")

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
    )

    base_model.trainable = True

    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D(name="image_features")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="sigmoid",
        name="multi_label_output",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="resnet50_from_scratch",
    )