from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.tsv"
TEST_PATH = ROOT / "data" / "test.tsv"

MODEL_NAME = "roberta-base"   # 先跑通；后面可换更强模型


def split_by_sentence_id(df, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["SentenceId"]
    tr_idx, va_idx = next(gss.split(df, groups=groups))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")

    tr_df, va_df = split_by_sentence_id(train_df)

    train_ds = Dataset.from_pandas(
        tr_df[["Phrase", "Sentiment"]].rename(columns={"Phrase": "text", "Sentiment": "label"}),
        preserve_index=False
    )

    eval_ds = Dataset.from_pandas(
        va_df[["Phrase", "Sentiment"]].rename(columns={"Phrase": "text", "Sentiment": "label"}),
        preserve_index=False
    )

    test_ds = Dataset.from_pandas(
        test_df[["PhraseId", "Phrase"]].rename(columns={"Phrase": "text"}),
        preserve_index=False
    )



    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        # batch["text"] 是一个 list，里面可能混有 NaN/None/非字符串
        texts = ["" if pd.isna(x) else str(x) for x in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=64)



    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tok, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "roberta_base"),
        learning_rate=2e-5,
        num_train_epochs=2,
        per_device_train_batch_size=4,     # 3050Ti 保险值
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,     # 等效 batch=16，显存友好
        fp16=True,                         # 混合精度：更快更省显存
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval:", metrics)

    # 预测 test 并生成提交
    test_logits = trainer.predict(test_ds).predictions
    test_pred = np.argmax(test_logits, axis=-1).astype(int)

    out_path = ROOT / "submissions" / "submission_roberta_base_gpu.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PhraseId": test_df["PhraseId"], "Sentiment": test_pred}).to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
