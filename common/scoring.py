# scoring.py
import math, torch
from typing import Iterable, List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FinBertScorer:
    """
    - model_dir: path
    - label_index: pick which logit/prob you consider "bullish" (0..num_labels-1)
      For common FinBERT (3 labels: negative, neutral, positive) -> bullish_idx=2
    - max_len: truncate safely; keep it fixed for predictable latency
    """
    def __init__(self, model_dir: str, bullish_idx: int = 2, max_len: int = 128, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.bullish_idx = bullish_idx
        self.max_len = max_len

    @torch.inference_mode()
    def score(self, items: Iterable[Dict], text_key: str = "text", batch_size: int = 32) -> List[Dict]:
        """
        items: iterable of dicts, each must have items[text_key]
               (you can also include article_id, etc., they’ll be forwarded)
        returns: list of dicts with original keys + score fields
        """
        buf, out = [], []
        append = out.append

        def flush():
            if not buf: return
            enc = self.tokenizer(
                [x[text_key] for x in buf],
                padding=True, truncation=True, max_length=self.max_len,
                return_tensors="pt", return_attention_mask=True
            )
            input_len = enc["input_ids"].shape[1]
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # [B, C]
            probs = torch.softmax(logits, dim=-1)  # [B, C]
            bull = probs[:, self.bullish_idx]      # bullish prob in [0,1]
            # score_raw: bullish prob; score_std: z-score via logit transform (optional)
            # logit(p) ~ N(0,1) if you standardize later; here we just give logit
            eps = 1e-6
            score_raw = bull
            score_std = torch.log((bull + eps) / (1 - bull + eps))  # logit

            for i, item in enumerate(buf):
                # detect truncation by checking if CLS..SEP filled to max_len and tokenizer flagged overflow
                # fast tokenizer doesn’t return overflow flag by default; practical proxy:
                truncated = len(self.tokenizer.tokenize(item[text_key])) > self.max_len - 2
                append({
                    **item,
                    "score_raw": float(score_raw[i].item()),
                    "score_std": float(score_std[i].item()),
                    "meta": {
                        "max_len": self.max_len,
                        "used_len": int(input_len),
                        "truncated": bool(truncated),
                    }
                })
            buf.clear()

        for it in items:
            buf.append(it)
            if len(buf) >= batch_size:
                flush()
        flush()
        return out
