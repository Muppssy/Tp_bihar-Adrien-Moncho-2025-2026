import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences


DEFAULT_THRESHOLD = 0.5


def parse_labels(label_string):
    return [int(label) for label in str(label_string).split()]


def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def label_distribution_from_binary(y_binary, classes):
    counts = y_binary.sum(axis=0)

    distribution = pd.DataFrame({
        "label": classes,
        "count": counts,
    })

    distribution["frequency_percent"] = (
        distribution["count"] / len(y_binary) * 100
    ).round(2)

    return distribution


def clean_text(text, stopwords_set=None, lemmatizer=None, min_token_length=3):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    if stopwords_set is not None:
        tokens = [token for token in tokens if token not in stopwords_set]

    tokens = [token for token in tokens if len(token) >= min_token_length]

    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def scores_to_binary_predictions(y_scores, threshold=DEFAULT_THRESHOLD):
    return (y_scores >= threshold).astype(int)


def evaluate_multilabel_model(
    model_name,
    y_true,
    y_scores,
    threshold=DEFAULT_THRESHOLD,
    y_pred=None,
):
    if y_pred is None:
        y_pred = scores_to_binary_predictions(y_scores, threshold=threshold)

    metrics = {
        "model": model_name,
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

    return metrics, y_pred


def compile_text_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )
    return model


def build_lstm_from_scratch(
    vocab_size,
    sequence_length,
    num_classes,
    embedding_dim=128,
    lstm_units=128,
    dropout_rate=0.4,
):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length,
        ),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ])

    return model


def build_bilstm_model(
    vocab_size,
    sequence_length,
    num_classes,
    embedding_dim=128,
    lstm_units=128,
    dropout_rate=0.4,
):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length,
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ])

    return model


def build_cnn1d_model(
    vocab_size,
    sequence_length,
    num_classes,
    embedding_dim=128,
    filters=128,
    kernel_size=3,
    dropout_rate=0.4,
):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length,
        ),
        tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ])

    return model


def build_hybrid_lstm_cnn1d_model(
    vocab_size,
    sequence_length,
    num_classes,
    embedding_dim=128,
    lstm_units=128,
    filters=128,
    kernel_size=3,
    dropout_rate=0.4,
):
    inputs = tf.keras.layers.Input(shape=(sequence_length,))

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
    )(inputs)

    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="hybrid_lstm_cnn1d",
    )


def build_glove_lstm_model(
    vocab_size,
    sequence_length,
    num_classes,
    embedding_matrix,
    embedding_dim,
    lstm_units=128,
    dropout_rate=0.4,
    trainable_embedding=False,
):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=sequence_length,
            trainable=trainable_embedding,
        ),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ])

    return model


def plot_multilabel_confusion_matrices_text(y_true, y_pred, class_names, ncols=6):
    cms = multilabel_confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    ncols = min(ncols, n_classes)
    nrows = int(np.ceil(n_classes / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (matrix, class_name) in enumerate(zip(cms, class_names)):
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


def build_model_comparison_table(metrics_list, sort_by="average_precision_macro"):
    comparison_df = pd.DataFrame(metrics_list)

    if sort_by in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by=sort_by, ascending=False)

    return comparison_df.reset_index(drop=True)


def plot_text_model_comparison(
    comparison_df,
    metrics=None,
    figsize=(12, 6),
    title="Comparaison des modeles texte sur le jeu de test",
):
    if metrics is None:
        metrics = [
            "f1_micro",
            "f1_macro",
            "average_precision_micro",
            "average_precision_macro",
            "auc_micro",
            "auc_macro",
        ]

    plot_df = comparison_df.set_index("model")[metrics]

    ax = plot_df.plot(kind="bar", figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel("Modele")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metrique", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def select_best_model_from_metrics(comparison_df, selection_metric="average_precision_macro"):
    if selection_metric not in comparison_df.columns:
        raise ValueError(f"Metrique de selection absente : {selection_metric}")

    best_row = comparison_df.sort_values(
        by=selection_metric,
        ascending=False,
    ).iloc[0]

    return best_row["model"], best_row.to_dict()


def active_labels_from_binary(binary_row, class_names):
    return [
        str(class_names[index])
        for index, value in enumerate(binary_row)
        if int(value) == 1
    ]


def top_label_scores(score_row, class_names, top_k=5):
    top_indices = np.argsort(score_row)[::-1][:top_k]
    return [
        f"{class_names[index]} ({score_row[index]:.3f})"
        for index in top_indices
    ]


def build_text_prediction_examples(
    test_df,
    y_true,
    y_scores,
    y_pred,
    class_names,
    n_examples=8,
    random_state=42,
):
    sample_df = test_df.reset_index(drop=True).sample(
        n=min(n_examples, len(test_df)),
        random_state=random_state,
    )

    rows = []

    for row_index, row in sample_df.iterrows():
        rows.append({
            "ImageID": row.get("ImageID", ""),
            "Caption": row["Caption"],
            "true_labels": active_labels_from_binary(y_true[row_index], class_names),
            "predicted_labels": active_labels_from_binary(y_pred[row_index], class_names),
            "top_scores": top_label_scores(y_scores[row_index], class_names),
        })

    return pd.DataFrame(rows)


def label_error_summary(y_true, y_pred, class_names):
    rows = []

    for index, class_name in enumerate(class_names):
        true_label = y_true[:, index]
        pred_label = y_pred[:, index]

        true_positives = int(((true_label == 1) & (pred_label == 1)).sum())
        false_positives = int(((true_label == 0) & (pred_label == 1)).sum())
        false_negatives = int(((true_label == 1) & (pred_label == 0)).sum())
        true_negatives = int(((true_label == 0) & (pred_label == 0)).sum())

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append({
            "label": str(class_name),
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "tn": true_negatives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

    return pd.DataFrame(rows)


def find_multilabel_error_examples(
    test_df,
    y_true,
    y_scores,
    y_pred,
    class_names,
    n_examples=10,
):
    test_df = test_df.reset_index(drop=True)
    error_counts = np.abs(y_true - y_pred).sum(axis=1)
    worst_indices = np.argsort(error_counts)[::-1][:n_examples]

    rows = []

    for index in worst_indices:
        false_negative_mask = (y_true[index] == 1) & (y_pred[index] == 0)
        false_positive_mask = (y_true[index] == 0) & (y_pred[index] == 1)

        rows.append({
            "ImageID": test_df.iloc[index].get("ImageID", ""),
            "Caption": test_df.iloc[index]["Caption"],
            "true_labels": active_labels_from_binary(y_true[index], class_names),
            "predicted_labels": active_labels_from_binary(y_pred[index], class_names),
            "false_negatives": [
                str(class_names[label_index])
                for label_index, value in enumerate(false_negative_mask)
                if value
            ],
            "false_positives": [
                str(class_names[label_index])
                for label_index, value in enumerate(false_positive_mask)
                if value
            ],
            "error_count": int(error_counts[index]),
            "top_scores": top_label_scores(y_scores[index], class_names),
        })

    return pd.DataFrame(rows)


def make_text_prediction_function(
    best_model_name,
    sklearn_models=None,
    sequence_models=None,
    tfidf_vectorizer=None,
    tokenizer=None,
    max_sequence_length=None,
    clean_text_kwargs=None,
):
    sklearn_models = sklearn_models or {}
    sequence_models = sequence_models or {}
    clean_text_kwargs = clean_text_kwargs or {}

    def predict_proba(raw_texts):
        cleaned_texts = [
            clean_text(text, **clean_text_kwargs)
            for text in raw_texts
        ]

        if best_model_name in sklearn_models:
            if tfidf_vectorizer is None:
                raise ValueError("tfidf_vectorizer est requis pour les modeles sklearn.")

            model = sklearn_models[best_model_name]
            features = tfidf_vectorizer.transform(cleaned_texts)

            if hasattr(model, "predict_proba"):
                return model.predict_proba(features)

            scores = model.decision_function(features)
            scores = np.clip(scores, -50, 50)
            return 1 / (1 + np.exp(-scores))

        if best_model_name in sequence_models:
            if tokenizer is None or max_sequence_length is None:
                raise ValueError(
                    "tokenizer et max_sequence_length sont requis pour les modeles sequentiels."
                )

            sequences = tokenizer.texts_to_sequences(cleaned_texts)
            padded = pad_sequences(
                sequences,
                maxlen=max_sequence_length,
                padding="post",
                truncating="post",
            )

            return sequence_models[best_model_name].predict(padded, verbose=0)

        raise ValueError(f"Modele inconnu pour la prediction LIME : {best_model_name}")

    return predict_proba


def explain_text_with_lime(
    text,
    predict_fn,
    class_names,
    label_index,
    num_features=10,
    num_samples=500,
):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=class_names)

    return explainer.explain_instance(
        text,
        predict_fn,
        labels=[label_index],
        num_features=num_features,
        num_samples=num_samples,
    )


def lime_explanation_to_dataframe(explanation, label_index):
    return pd.DataFrame(
        explanation.as_list(label=label_index),
        columns=["word", "weight"],
    )


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _slugify_model_name(model_name):
    return (
        str(model_name)
        .lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("-", "_")
    )


def save_best_text_artifacts(
    best_model_name,
    model,
    model_type,
    registry_dir,
    classes,
    metrics=None,
    tokenizer=None,
    tfidf_vectorizer=None,
    max_sequence_length=None,
    vocab_size=None,
):
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_slug = _slugify_model_name(best_model_name)

    if model_type == "keras":
        model_path = registry_dir / "text_model.keras"
        model.save(model_path)
    elif model_type == "sklearn":
        model_path = registry_dir / f"text_model_{model_slug}.joblib"
        joblib.dump(model, model_path)
    else:
        raise ValueError("model_type doit etre egal a 'keras' ou 'sklearn'.")

    tokenizer_path = None
    tfidf_vectorizer_path = None

    if tokenizer is not None:
        tokenizer_path = registry_dir / "text_tokenizer.joblib"
        joblib.dump(tokenizer, tokenizer_path)

    if tfidf_vectorizer is not None:
        tfidf_vectorizer_path = registry_dir / "text_tfidf_vectorizer.joblib"
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)

    metadata = {
        "model_name": best_model_name,
        "model_type": model_type,
        "model_path": str(model_path),
        "classes": [str(class_name) for class_name in classes],
        "metrics": {
            key: _json_safe(value)
            for key, value in (metrics or {}).items()
        },
        "max_sequence_length": _json_safe(max_sequence_length),
        "vocab_size": _json_safe(vocab_size),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "tfidf_vectorizer_path": (
            str(tfidf_vectorizer_path) if tfidf_vectorizer_path else None
        ),
    }

    metadata_path = registry_dir / "text_model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "tokenizer_path": tokenizer_path,
        "tfidf_vectorizer_path": tfidf_vectorizer_path,
    }
