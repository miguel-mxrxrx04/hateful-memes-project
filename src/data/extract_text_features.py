import os
import json
from pathlib import Path
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

def extract_text_features(batch_size: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    model.eval()

    raw_dir = Path("data/raw")
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "dev", "test"]

    for split in splits:
        jsonl_path = raw_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"File {jsonl_path} does not exist. Skipping.")
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line.strip()) for line in f if line.strip()]

        for i in tqdm(range(0, len(samples), batch_size), desc=f"Processing {split}"):
            batch_samples = samples[i:i + batch_size]
            texts = [sample["text"] for sample in batch_samples]
            meme_ids = [str(sample["id"]).zfill(5) for sample in batch_samples]

            # Tokenize and encode the text
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # Use the [CLS] token representation

            # Save embeddings to disk
            for meme_id, embedding in zip(meme_ids, embeddings):
                feat_path = features_dir / f"{meme_id}.pt"
                torch.save(embedding, feat_path)
    print("Text feature extraction completed.")
if __name__ == "__main__":
    extract_text_features()