# scoring_regression.py
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "training" / "data" / "artifacts" / "stage1_from_splits" / "best"
MAX_LEN = 64
BATCH = 64

# load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print(Path(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(Path(MODEL_DIR))
model.eval()

def _text_from_article(a: dict) -> str:
    # choose your text fields consistently with training
    title = a.get("title") or ""
    desc  = a.get("description") or ""
    txt = (title + " " + desc).strip()
    return txt if txt else title or desc

def score_articles_regression(articles: list[dict]) -> list[dict]:
    texts = [_text_from_article(a) for a in articles]
    out = []

    with torch.inference_mode():
        for i in range(0, len(texts), BATCH):
            batch_txt = texts[i:i+BATCH]
            enc = tokenizer(
                batch_txt,
                truncation=True, padding=True, max_length=MAX_LEN,
                return_tensors="pt"
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits           # shape [N, 1]
            scores = logits.squeeze(1).cpu().tolist()  # list of floats

            for j, s in enumerate(scores):
                idx = i + j
                out.append({
                    "id": articles[idx].get("id"),
                    "provider_id": articles[idx].get("id"),
                    "score_raw": float(s),
                })
    return out
