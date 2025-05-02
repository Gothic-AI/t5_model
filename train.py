import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from sklearn.model_selection import train_test_split

# === Config ===
model_name = "t5-small"
max_input_len = 64
max_target_len = 64
batch_size = 8
epochs = 3
model_path = "./files/model/t5_dracula"
train_csv = "./files/data/dracula_train.csv"
val_csv = "./files/data/dracula_val.csv"

# === Tokenizer and Model ===
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

# === Load and Prepare Dataset ===
def encode_data(df):
    input_enc = tokenizer(
        df["input"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_input_len,
        return_tensors="tf"
    )
    target_enc = tokenizer(
        df["target"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_target_len,
        return_tensors="tf"
    )
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "decoder_input_ids": target_enc["input_ids"]
    }, target_enc["input_ids"]

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return tf.data.Dataset.from_tensor_slices(encode_data(df))

# === Build Train Function ===
def train():
    train_dataset = load_dataset(train_csv).shuffle(100).batch(batch_size)
    val_dataset = load_dataset(val_csv).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer)  # Default loss is fine for seq2seq

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                save_best_only=True,
                save_weights_only=False
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True
            )
        ]
    )
    # ✅ Save model and tokenizer in Hugging Face format
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("✅ Model and tokenizer saved.")

# === Build Prediction Function ===
def predict():
    df = pd.read_csv("./files/data/dracula_test.csv")  # Replace with your test file path
    inputs = tokenizer(
        df["input"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_input_len,
        return_tensors="tf"
    )

    model = TFT5ForConditionalGeneration.from_pretrained(model_path)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_target_len
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    df["generated_answer"] = decoded
    df.to_csv("./files/data/test_predictions.csv", index=False)
    print("✅ Predictions saved to test_predictions.csv")

# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "predict"])
    args = parser.parse_args()
    globals()[args.command]()
