import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = DATA_DIR / "submission.csv"


def flatten_turns(value) -> str:
    """Convert a serialized list like '["q1","q2"]' into a single text block."""
    if pd.isna(value):
        return ""
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return "\n".join(str(x) for x in parsed)
        except Exception:
            pass
    return text


def build_pair_text(df: pd.DataFrame) -> pd.Series:
    prompt = df["prompt"].map(flatten_turns)
    resp_a = df["response_a"].map(flatten_turns)
    resp_b = df["response_b"].map(flatten_turns)
    return (
        "[PROMPT]\n"
        + prompt
        + "\n\n[RESPONSE_A]\n"
        + resp_a
        + "\n\n[RESPONSE_B]\n"
        + resp_b
    )


def onehot_to_class_index(df: pd.DataFrame) -> pd.Series:
    cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    return df[cols].values.argmax(axis=1)


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print("Building model text...")
    x_all = build_pair_text(train_df)
    x_test = build_pair_text(test_df)
    y_all = onehot_to_class_index(train_df)

    print("Creating validation split...")
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all,
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=150000,
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_valid_vec = vectorizer.transform(x_valid)

    print("Training multinomial logistic regression...")
    clf = LogisticRegression(
        solver="saga",
        max_iter=250,
        C=3.0,
        random_state=42,
    )
    clf.fit(x_train_vec, y_train)

    valid_proba = clf.predict_proba(x_valid_vec)
    score = log_loss(y_valid, valid_proba, labels=[0, 1, 2])
    print(f"Validation LogLoss: {score:.6f}")

    print("Refitting on all train data...")
    x_all_vec = vectorizer.fit_transform(x_all)
    x_test_vec = vectorizer.transform(x_test)
    clf.fit(x_all_vec, y_all)
    test_proba = clf.predict_proba(x_test_vec)

    print("Building submission...")
    sub = sample_sub.copy()
    sub["winner_model_a"] = test_proba[:, 0]
    sub["winner_model_b"] = test_proba[:, 1]
    sub["winner_tie"] = test_proba[:, 2]
    sub.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
