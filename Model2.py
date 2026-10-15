import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    Layer
)
from tensorflow.keras.layers import TextVectorization

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

np.random.seed(42)
tf.random.set_seed(42)

print("Libraries imported successfully!")
print("TensorFlow version:", tf.__version__)
df = pd.read_csv("MLdatasetfinal.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
texts = df["text_"].values

y = df["label_binary"].values.astype("float32")

numeric_data = df[
    [
        "rating",
        "text_length",
        "sentence_count",
        "avg_word_len",
        "vocab_richness"
    ]
].values.astype("float32")

print("Text samples:", len(texts))
print("Numeric feature shape:", numeric_data.shape)
print("Target shape:", y.shape)
category_encoded = pd.get_dummies(
    df["category"],
    prefix="category",
    dtype=float
)

category_data = category_encoded.values.astype(
    "float32"
)

print("Category feature shape:", category_data.shape)
print("\nCategories:")
print(df["category"].unique())
(
    text_train,
    text_test,
    numeric_train,
    numeric_test,
    category_train,
    category_test,
    y_train,
    y_test
) = train_test_split(
    texts,
    numeric_data,
    category_data,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(text_train))
print("Testing samples:", len(text_test))
# ============================================================
# STEP 14 ONWARDS
# HYBRID BiLSTM + ATTENTION + LINGUISTIC FEATURES
# ============================================================


# ============================================================
# STEP 14 — SCALE NUMERICAL FEATURES
# ============================================================

scaler = StandardScaler()

numeric_train_scaled = scaler.fit_transform(
    numeric_train
).astype("float32")

numeric_test_scaled = scaler.transform(
    numeric_test
).astype("float32")

print("=" * 60)
print("STEP 14 COMPLETED")
print("=" * 60)
print("Numerical features scaled successfully!")

print("\nTraining numeric shape:",
      numeric_train_scaled.shape)

print("Testing numeric shape:",
      numeric_test_scaled.shape)


# ============================================================
# STEP 15 — TEXT VECTORIZATION
# ============================================================

MAX_WORDS = 10000
SEQUENCE_LENGTH = 200

vectorizer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation"
)

# Learn vocabulary ONLY from training text
vectorizer.adapt(
    tf.data.Dataset
    .from_tensor_slices(text_train)
    .batch(32)
)

vocabulary_size = len(
    vectorizer.get_vocabulary()
)

print("\n" + "=" * 60)
print("STEP 15 COMPLETED")
print("=" * 60)

print("Vocabulary size:",
      vocabulary_size)


# ============================================================
# STEP 16 — CONVERT TEXT INTO NUMERICAL SEQUENCES
# ============================================================

X_train_text = vectorizer(
    tf.constant(text_train)
)

X_test_text = vectorizer(
    tf.constant(text_test)
)

print("\n" + "=" * 60)
print("STEP 16 COMPLETED")
print("=" * 60)

print("Training text shape:",
      X_train_text.shape)

print("Testing text shape:",
      X_test_text.shape)


# ============================================================
# STEP 17 — CREATE ATTENTION LAYER
# ============================================================

class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super(
            AttentionLayer,
            self
        ).build(input_shape)

    def call(self, inputs):

        score = tf.tanh(
            tf.matmul(
                inputs,
                self.W
            ) + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = (
            inputs *
            attention_weights
        )

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector


print("\n" + "=" * 60)
print("STEP 17 COMPLETED")
print("=" * 60)

print("Attention layer created successfully!")


# ============================================================
# STEP 18 — CREATE TEXT BRANCH
# ============================================================

text_input = Input(
    shape=(SEQUENCE_LENGTH,),
    name="text_input"
)

embedding = Embedding(
    input_dim=vocabulary_size,
    output_dim=128,
    name="word_embedding"
)(text_input)

bilstm = Bidirectional(
    LSTM(
        64,
        return_sequences=True,
        dropout=0.2
    ),
    name="bidirectional_lstm"
)(embedding)

attention = AttentionLayer(
    name="attention"
)(bilstm)

text_features = Dense(
    64,
    activation="relu",
    name="text_features"
)(attention)

text_features = Dropout(
    0.3
)(text_features)

print("\n" + "=" * 60)
print("STEP 18 COMPLETED")
print("=" * 60)

print("Text branch created successfully!")


# ============================================================
# STEP 19 — CREATE LINGUISTIC FEATURE BRANCH
# ============================================================

numeric_input = Input(
    shape=(5,),
    name="numeric_input"
)

numeric_features = Dense(
    32,
    activation="relu",
    name="linguistic_features"
)(numeric_input)

numeric_features = Dropout(
    0.2
)(numeric_features)

print("\n" + "=" * 60)
print("STEP 19 COMPLETED")
print("=" * 60)

print("Linguistic feature branch created!")


# ============================================================
# STEP 20 — CREATE CATEGORY BRANCH
# ============================================================

category_input = Input(
    shape=(category_train.shape[1],),
    name="category_input"
)

category_features = Dense(
    16,
    activation="relu",
    name="category_features"
)(category_input)

print("\n" + "=" * 60)
print("STEP 20 COMPLETED")
print("=" * 60)

print("Category branch created!")


# ============================================================
# STEP 21 — COMBINE ALL FEATURES
# ============================================================

combined = Concatenate(
    name="feature_fusion"
)([
    text_features,
    numeric_features,
    category_features
])

print("\n" + "=" * 60)
print("STEP 21 COMPLETED")
print("=" * 60)

print("Text + Linguistic + Category features combined!")


# ============================================================
# STEP 22 — FINAL DENSE LAYERS
# ============================================================

combined = Dense(
    64,
    activation="relu",
    name="fusion_dense"
)(combined)

combined = Dropout(
    0.4
)(combined)

combined = Dense(
    32,
    activation="relu"
)(combined)

combined = Dropout(
    0.3
)(combined)

print("\n" + "=" * 60)
print("STEP 22 COMPLETED")
print("=" * 60)

print("Final dense layers created!")


# ============================================================
# STEP 23 — OUTPUT LAYER
# ============================================================

output = Dense(
    1,
    activation="sigmoid",
    name="prediction"
)(combined)

print("\n" + "=" * 60)
print("STEP 23 COMPLETED")
print("=" * 60)

print("Sigmoid output layer created!")


# ============================================================
# STEP 24 — BUILD MODEL
# ============================================================

model = Model(
    inputs=[
        text_input,
        numeric_input,
        category_input
    ],
    outputs=output
)

print("\n" + "=" * 60)
print("STEP 24 COMPLETED")
print("=" * 60)

print("Hybrid BiLSTM model created successfully!")


# ============================================================
# STEP 25 — COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

print("\n" + "=" * 60)
print("STEP 25 COMPLETED")
print("=" * 60)

print("Model compiled successfully!")


# ============================================================
# STEP 26 — DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\n" + "=" * 60)
print("STEP 26 — MODEL ARCHITECTURE")
print("=" * 60)

model.summary()


# ============================================================
# STEP 27 — CREATE CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

print("\n" + "=" * 60)
print("STEP 27 COMPLETED")
print("=" * 60)

print("Callbacks created successfully!")


# ============================================================
# STEP 28 — TRAIN THE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 28 — TRAINING STARTED")
print("=" * 60)

print("Please wait...")
print("Do NOT click Run repeatedly while training.")

history = model.fit(

    [
        X_train_text,
        numeric_train_scaled,
        category_train
    ],

    y_train,

    validation_split=0.20,

    epochs=30,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)


# ============================================================
# STEP 29 — EVALUATE MODEL
# ============================================================

results = model.evaluate(
    [
        X_test_text,
        numeric_test_scaled,
        category_test
    ],

    y_test,

    verbose=0
)

print("\n" + "=" * 60)
print("STEP 29 — MODEL EVALUATION")
print("=" * 60)

print(
    f"Test Loss      : {results[0]:.4f}"
)

print(
    f"Test Accuracy  : {results[1]:.4f}"
)

print(
    f"Test Precision : {results[2]:.4f}"
)

print(
    f"Test Recall    : {results[3]:.4f}"
)


# ============================================================
# STEP 30 — GENERATE PREDICTIONS
# ============================================================

y_probability = model.predict(
    [
        X_test_text,
        numeric_test_scaled,
        category_test
    ],

    verbose=0
).flatten()

y_pred = (
    y_probability >= 0.5
).astype(int)

print("\n" + "=" * 60)
print("STEP 30 COMPLETED")
print("=" * 60)

print("Predictions generated!")
print("Number of predictions:",
      len(y_pred))


# ============================================================
# STEP 31 — PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("STEP 31 — FINAL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# STEP 32 — CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("STEP 32 — CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake (CG)",
            "Real (OR)"
        ]
    )
)


# ============================================================
# STEP 33 — CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",

    xticklabels=[
        "Fake (CG)",
        "Real (OR)"
    ],

    yticklabels=[
        "Fake (CG)",
        "Real (OR)"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Hybrid BiLSTM - Confusion Matrix"
)

plt.show()


# ============================================================
# STEP 34 — TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.show()


# ============================================================
# STEP 35 — TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.show()


# ============================================================
# STEP 36 — NEW REVIEW PREDICTION FUNCTION
# ============================================================

def predict_new_review(
    review,
    rating=5,
    category=None
):

    # ----------------------------------------
    # Calculate text features
    # ----------------------------------------

    words = review.split()

    text_length = len(review)

    sentence_count = max(
        1,
        review.count(".")
        + review.count("!")
        + review.count("?")
    )

    if len(words) > 0:

        avg_word_len = np.mean(
            [
                len(word)
                for word in words
            ]
        )

        vocab_richness = (
            len(
                set(
                    word.lower()
                    for word in words
                )
            )
            / len(words)
        )

    else:

        avg_word_len = 0

        vocab_richness = 0


    # ----------------------------------------
    # Category
    # ----------------------------------------

    if category is None:

        category = df[
            "category"
        ].mode()[0]


    # ----------------------------------------
    # Create category vector
    # ----------------------------------------

    category_vector = np.zeros(
        category_encoded.shape[1],
        dtype="float32"
    )

    category_columns = (
        category_encoded.columns.tolist()
    )

    column_name = (
        "category_" + str(category)
    )

    if column_name in category_columns:

        index = category_columns.index(
            column_name
        )

        category_vector[index] = 1.0


    # ----------------------------------------
    # Numeric features
    # ----------------------------------------

    numeric_vector = np.array([

        rating,

        text_length,

        sentence_count,

        avg_word_len,

        vocab_richness

    ]).reshape(1, -1)


    numeric_vector = scaler.transform(
        numeric_vector
    ).astype("float32")


    # ----------------------------------------
    # Convert review into sequence
    # ----------------------------------------

    text_vector = vectorizer(
        tf.constant([review])
    )


    # ----------------------------------------
    # Predict
    # ----------------------------------------

    probability = model.predict(
        [
            text_vector,

            numeric_vector,

            category_vector.reshape(
                1,
                -1
            )
        ],

        verbose=0
    )[0][0]


    # ----------------------------------------
    # Classification
    # ----------------------------------------

    if probability >= 0.5:

        prediction = (
            "REAL / ORIGINAL (OR)"
        )

        confidence = probability

    else:

        prediction = (
            "FAKE / COMPUTER GENERATED (CG)"
        )

        confidence = 1 - probability


    print("\n" + "=" * 60)

    print("NEW REVIEW PREDICTION")

    print("=" * 60)

    print("\nReview:")
    print(review)

    print("\nPrediction:")
    print(prediction)

    print(
        f"\nProbability of REAL: "
        f"{probability:.4f}"
    )

    print(
        f"Estimated confidence: "
        f"{confidence:.4f}"
    )

    print("=" * 60)

    return prediction


# ============================================================
# STEP 37 — TEST A NEW REVIEW
# ============================================================

new_review = input(
    "\nEnter a review to test: "
)

predict_new_review(
    review=new_review,
    rating=5
)
