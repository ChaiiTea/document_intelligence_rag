"""Check what labels the model actually uses internally."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
print("Model id2label:", model.config.id2label)
print("Model label2id:", model.config.label2id)
