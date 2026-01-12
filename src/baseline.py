from pathlib import Path
import pandas as pd
from sklearn.pipeline import FeatureUnion
import numpy as np
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.tsv"
TEST_PATH = DATA_DIR / "test.tsv"

from sklearn.model_selection import GroupShuffleSplit

def split_by_sentence_id(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["SentenceId"]
    train_idx, val_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def main():
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")

    train_part, val_part = split_by_sentence_id(train_df)

    X_train = train_part["Phrase"].astype(str)
    y_train = train_part["Sentiment"].astype(int)

    X_val = val_part["Phrase"].astype(str)
    y_val = val_part["Sentiment"].astype(int)

    vectorizer = FeatureUnion(
        [
            ("word", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=200000
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2
            )),
        ],
        transformer_weights={"word": 0.5, "char": 1.0}
    )


    model = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1
        ))
    ])


    Cs = [0.25, 0.5, 1.0, 2.0]

    best_acc = -1
    best_C = None
    best_model = None

    for C in Cs:
        print(f"\n[Train] start C={C}", flush=True)

        model = Pipeline([
            ("tfidf", vectorizer),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=C
            ))
        ])

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        print(f"C={C:<4}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_C = C
            best_model = model

    print(f"\nBEST: C={best_C}  val_acc={best_acc:.4f}")


    # 生成提交文件
    test_pred = best_model.predict(test_df["Phrase"].astype(str))


    sub_df = pd.DataFrame({
        "PhraseId": test_df["PhraseId"],
        "Sentiment": test_pred.astype(int)
    })

    out_path = ROOT / "submissions" / "submission_union_weight_word1_char0p5.csv"


    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
