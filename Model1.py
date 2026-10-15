# ============================================================
# FAKE REVIEW DETECTION USING LINEAR SVM
# TF-IDF + LINGUISTIC FEATURES
# ============================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================
# 2. LOAD DATASET
# =========================

# If your dataset is CSV:
df = pd.read_csv("MLdatasetfinal.csv")

# If your dataset is Excel, comment the above line and use:
# df = pd.read_excel("MLdatasetfinal.xlsx")


print("=" * 60)
print("FAKE REVIEW DETECTION PROJECT")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# =========================
# 3. BASIC DATASET CHECK
# =========================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Reviews:")
print(df["text_"].duplicated().sum())


# =========================
# 4. REMOVE MISSING VALUES
# =========================

# Remove rows where important columns are missing

required_columns = [
    "text_",
    "label_binary",
    "rating",
    "text_length",
    "sentence_count",
    "avg_word_len",
    "vocab_richness"
]

df = df.dropna(subset=required_columns).copy()

print("\nDataset shape after removing missing values:")
print(df.shape)


# =========================
# 5. REMOVE DUPLICATE REVIEWS
# =========================

df = df.drop_duplicates(subset=["text_"]).copy()

print("\nDataset shape after removing duplicate reviews:")
print(df.shape)


# =========================
# 6. LABEL DISTRIBUTION
# =========================

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

print("\nOriginal Labels:")
print(df["label"].value_counts())

print("\nBinary Labels:")
print(df["label_binary"].value_counts())


# =========================
# 7. LABEL MEANING
# =========================

print("\nLabel Mapping:")
print("0 = CG = Computer Generated / Fake")
print("1 = OR = Original / Real")


# =========================
# 8. FEATURE ANALYSIS
# =========================

linguistic_features = [
    "text_length",
    "sentence_count",
    "avg_word_len",
    "vocab_richness"
]

print("\n" + "=" * 60)
print("AVERAGE FEATURES BY LABEL")
print("=" * 60)

print(
    df.groupby("label")[
        [
            "rating",
            "text_length",
            "sentence_count",
            "avg_word_len",
            "vocab_richness"
        ]
    ].mean()
)


# =========================
# 9. VISUALIZE LABELS
# =========================

plt.figure(figsize=(6, 4))

df["label"].value_counts().plot(kind="bar")

plt.title("Fake vs Real Reviews")
plt.xlabel("Review Type")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.show()


# =========================
# 10. VISUALIZE LINGUISTIC FEATURES
# =========================

for feature in linguistic_features:

    plt.figure(figsize=(6, 4))

    df.groupby("label")[feature].mean().plot(kind="bar")

    plt.title(f"{feature} by Review Type")
    plt.xlabel("Review Type")
    plt.ylabel(feature)
    plt.xticks(rotation=0)

    plt.show()


# ============================================================
# 11. PREPARE INPUT AND TARGET
# ============================================================

X_text = df["text_"]
y = df["label_binary"]


# ============================================================
# 12. TRAIN-TEST SPLIT
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))


# ============================================================
# 13. TF-IDF FEATURE EXTRACTION
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train_text)

X_test_tfidf = tfidf.transform(X_test_text)

print("\n" + "=" * 60)
print("TF-IDF INFORMATION")
print("=" * 60)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ============================================================
# 14. MODEL 1 - TF-IDF + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1: TF-IDF + LINEAR SVM")
print("=" * 60)

baseline_svm = LinearSVC(random_state=42)

baseline_svm.fit(
    X_train_tfidf,
    y_train
)

y_pred_baseline = baseline_svm.predict(
    X_test_tfidf
)

baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)

baseline_precision = precision_score(
    y_test,
    y_pred_baseline
)

baseline_recall = recall_score(
    y_test,
    y_pred_baseline
)

baseline_f1 = f1_score(
    y_test,
    y_pred_baseline
)

print("\nAccuracy:", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall:", baseline_recall)
print("F1 Score:", baseline_f1)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_baseline,
        target_names=["Fake (CG)", "Real (OR)"]
    )
)


# ============================================================
# 15. CONFUSION MATRIX - MODEL 1
# ============================================================

cm1 = confusion_matrix(
    y_test,
    y_pred_baseline
)

disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm1,
    display_labels=["Fake (CG)", "Real (OR)"]
)

disp1.plot()

plt.title("Confusion Matrix - TF-IDF + Linear SVM")
plt.show()


# ============================================================
# 16. MODEL 2 - LINGUISTIC FEATURES + LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2: LINGUISTIC FEATURES + LINEAR SVM")
print("=" * 60)

X_ling = df[linguistic_features]

X_train_ling, X_test_ling, y_train_ling, y_test_ling = train_test_split(
    X_ling,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_ling_scaled = scaler.fit_transform(
    X_train_ling
)

X_test_ling_scaled = scaler.transform(
    X_test_ling
)

ling_svm = LinearSVC(random_state=42)

ling_svm.fit(
    X_train_ling_scaled,
    y_train_ling
)

y_pred_ling = ling_svm.predict(
    X_test_ling_scaled
)

ling_accuracy = accuracy_score(
    y_test_ling,
    y_pred_ling
)

ling_precision = precision_score(
    y_test_ling,
    y_pred_ling
)

ling_recall = recall_score(
    y_test_ling,
    y_pred_ling
)

ling_f1 = f1_score(
    y_test_ling,
    y_pred_ling
)

print("\nAccuracy:", ling_accuracy)
print("Precision:", ling_precision)
print("Recall:", ling_recall)
print("F1 Score:", ling_f1)

print("\nClassification Report:")

print(
    classification_report(
        y_test_ling,
        y_pred_ling,
        target_names=["Fake (CG)", "Real (OR)"]
    )
)


# ============================================================
# 17. MODEL 3 - TF-IDF + LINGUISTIC FEATURES + SVM
# ============================================================

print("\n" + "=" * 60)
print("MODEL 3: TF-IDF + LINGUISTIC FEATURES + LINEAR SVM")
print("=" * 60)

# Create matching train/test splits

X_text_train, X_text_test, X_ling_train, X_ling_test, y_train2, y_test2 = train_test_split(
    df["text_"],
    df[linguistic_features],
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF

tfidf_proposed = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf2 = tfidf_proposed.fit_transform(
    X_text_train
)

X_test_tfidf2 = tfidf_proposed.transform(
    X_text_test
)


# Scale linguistic features

scaler_proposed = StandardScaler()

X_train_ling_scaled2 = scaler_proposed.fit_transform(
    X_ling_train
)

X_test_ling_scaled2 = scaler_proposed.transform(
    X_ling_test
)


# Combine TF-IDF + linguistic features

X_train_combined = hstack([
    X_train_tfidf2,
    X_train_ling_scaled2
])

X_test_combined = hstack([
    X_test_tfidf2,
    X_test_ling_scaled2
])


print("\nCombined Training Shape:")
print(X_train_combined.shape)

print("\nCombined Testing Shape:")
print(X_test_combined.shape)


# Train SVM

proposed_svm = LinearSVC(random_state=42)

proposed_svm.fit(
    X_train_combined,
    y_train2
)

y_pred_proposed = proposed_svm.predict(
    X_test_combined
)


# Evaluate

proposed_accuracy = accuracy_score(
    y_test2,
    y_pred_proposed
)

proposed_precision = precision_score(
    y_test2,
    y_pred_proposed
)

proposed_recall = recall_score(
    y_test2,
    y_pred_proposed
)

proposed_f1 = f1_score(
    y_test2,
    y_pred_proposed
)

print("\nAccuracy:", proposed_accuracy)
print("Precision:", proposed_precision)
print("Recall:", proposed_recall)
print("F1 Score:", proposed_f1)

print("\nClassification Report:")

print(
    classification_report(
        y_test2,
        y_pred_proposed,
        target_names=["Fake (CG)", "Real (OR)"]
    )
)


# ============================================================
# 18. CONFUSION MATRIX - PROPOSED MODEL
# ============================================================

cm3 = confusion_matrix(
    y_test2,
    y_pred_proposed
)

disp3 = ConfusionMatrixDisplay(
    confusion_matrix=cm3,
    display_labels=["Fake (CG)", "Real (OR)"]
)

disp3.plot()

plt.title("Confusion Matrix - Proposed Model")
plt.show()


# ============================================================
# 19. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "TF-IDF + Linear SVM",
        "Linguistic Features + Linear SVM",
        "TF-IDF + Linguistic Features + Linear SVM"
    ],

    "Accuracy": [
        baseline_accuracy,
        ling_accuracy,
        proposed_accuracy
    ],

    "Precision": [
        baseline_precision,
        ling_precision,
        proposed_precision
    ],

    "Recall": [
        baseline_recall,
        ling_recall,
        proposed_recall
    ],

    "F1 Score": [
        baseline_f1,
        ling_f1,
        proposed_f1
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results)


# ============================================================
# 20. MODEL ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.ylim(0, 1)

plt.tight_layout()

plt.show()


# ============================================================
# 21. FEATURE IMPORTANCE - TF-IDF MODEL
# ============================================================

print("\n" + "=" * 60)
print("TOP TF-IDF FEATURES")
print("=" * 60)

feature_names = tfidf.get_feature_names_out()

coefficients = baseline_svm.coef_[0]

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": coefficients

})


# Features associated with Real / OR

top_real = feature_importance.sort_values(
    "Coefficient",
    ascending=False
).head(20)

print("\nTop features associated with Real (OR):")

print(top_real)


# Features associated with Fake / CG

top_fake = feature_importance.sort_values(
    "Coefficient",
    ascending=True
).head(20)

print("\nTop features associated with Fake (CG):")

print(top_fake)


# ============================================================
# 22. 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

svm_pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
    ),

    (
        "svm",
        LinearSVC(random_state=42)
    )

])

cv_scores = cross_val_score(
    svm_pipeline,
    df["text_"],
    y,
    cv=cv,
    scoring="f1"
)

print("\n5-Fold F1 Scores:")

print(cv_scores)

print("\nMean F1 Score:")

print(cv_scores.mean())

print("\nStandard Deviation:")

print(cv_scores.std())


# ============================================================
# 23. FINAL MODEL FOR PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("FINAL FAKE REVIEW PREDICTION SYSTEM")
print("=" * 60)


def predict_review(review):

    review_tfidf = tfidf.transform(
        [review]
    )

    prediction = baseline_svm.predict(
        review_tfidf
    )[0]

    decision_score = baseline_svm.decision_function(
        review_tfidf
    )[0]

    if prediction == 0:

        result = "FAKE / COMPUTER GENERATED (CG)"

    else:

        result = "REAL / ORIGINAL (OR)"

    return result, decision_score


# ============================================================
# 24. TEST A NEW REVIEW
# ============================================================

new_review = input(
    "\nEnter a review to check whether it is Fake or Real: "
)

prediction, score = predict_review(
    new_review
)

print("\n" + "=" * 60)

print("REVIEW:")
print(new_review)

print("\nPREDICTION:")
print(prediction)

print("\nDECISION SCORE:")
print(score)

print("=" * 60)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("\nDataset Size:", df.shape)

print("\nTarget:")
print("0 = CG = Fake / Computer Generated")
print("1 = OR = Real / Original")

print("\nModel Results:")

print(results)

print("\nBest accuracy among tested models:")

best_model = results.loc[
    results["Accuracy"].idxmax()
]

print(best_model["Model"])

print(
    "Accuracy:",
    best_model["Accuracy"]
)

print("\nProject Objective:")
print(
    "To detect whether a review is Fake/Computer Generated "
    "or Real/Original using machine learning."
)

print("\nMain Algorithm:")
print("Linear SVM")

print("\nText Feature Extraction:")
print("TF-IDF")

print("\nAdditional Linguistic Features:")
print(
    "Text Length, Sentence Count, Average Word Length, "
    "Vocabulary Richness"
)

print("\nProject Completed Successfully!")
print("=" * 60)
