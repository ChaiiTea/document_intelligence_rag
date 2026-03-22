from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
FUNSD_TEST = Path("data/annotations/funsd/test.json")
 
# FUNSD integer label mapping (from nielsr/funsd-layoutlmv3)
ID2LABEL = {0: "O", 1: "B-HEADER", 2: "I-HEADER", 3: "B-QUESTION", 4: "I-QUESTION", 5: "B-ANSWER", 6: "I-ANSWER"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
# Also keep stripped labels as fallback
LABEL2ID.update({"ANSWER": 5, "QUESTION": 3, "HEADER": 1})
LABEL_LIST = [ID2LABEL[i] for i in sorted(ID2LABEL)]
 
 
def load_funsd_test() -> list[dict]:
    if not FUNSD_TEST.exists():
        print(f"ERROR: {FUNSD_TEST} not found. Run: python scripts/prepare_funsd.py")
        sys.exit(1)
    with open(FUNSD_TEST) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test documents from {FUNSD_TEST}")
 
    # Normalise field names — dataset uses 'tokens', we want 'words'
    for ex in data:
        if "words" not in ex or not ex["words"]:
            ex["words"] = ex.get("tokens", [])
    return data
 
 
def baseline_predict(example: dict) -> list[int]:
    """Predict O for every token — naive baseline."""
    return [LABEL2ID["O"]] * len(example["words"])
 
 
def model_predict(example: dict, extractor) -> list[int]:
    from src.ocr.pdf_loader import PageTokens, Token
 
    words = example["words"]
    bboxes = example["bboxes"]
 
    # Keep a flat index→token mapping so we can map spans back reliably
    tokens = []
    for word, bbox in zip(words, bboxes):
        tokens.append(Token(
            text=word,
            bbox=(bbox[0]/1000, bbox[1]/1000, bbox[2]/1000, bbox[3]/1000),
            confidence=1.0,
            page=0,
        ))
 
    page = PageTokens(page_number=0, width=1000, height=1000, tokens=tokens)
 
    # Build a lookup: token object id → index in our list
    token_id_to_idx = {id(t): i for i, t in enumerate(tokens)}
 
    result = extractor.extract([page], pdf_name=str(example.get("id", "")))
 
    # Map spans back to per-token labels using object identity (no bbox float drift)
    token_labels = ["O"] * len(tokens)
    for span in result.spans:
        label = span.label
        for tok in span.tokens:
            idx = token_id_to_idx.get(id(tok))
            if idx is not None:
                token_labels[idx] = label
 
    return [LABEL2ID.get(l, 0) for l in token_labels]
 
 
def evaluate():
    from sklearn.metrics import f1_score, classification_report
    import mlflow
    from src.extraction.layoutlm import LayoutLMExtractor
 
    examples = load_funsd_test()
 
    # Quick sanity check on label types
    sample_labels = examples[0]["labels"]
    print(f"Sample labels (first 5): {sample_labels[:5]}")
    print(f"Label type: {type(sample_labels[0])}")
 
    extractor = LayoutLMExtractor()
 
    y_true_all, y_baseline_all, y_model_all = [], [], []
 
    for i, ex in enumerate(examples):
        print(f"Processing doc {i+1}/{len(examples)}: {ex.get('id', '')} ({len(ex['words'])} tokens)")
 
        true_labels = ex["labels"]  # these are already ints from HuggingFace
        baseline_preds = baseline_predict(ex)
        model_preds = model_predict(ex, extractor)
 
        min_len = min(len(true_labels), len(baseline_preds), len(model_preds))
        y_true_all.extend(true_labels[:min_len])
        y_baseline_all.extend(baseline_preds[:min_len])
        y_model_all.extend(model_preds[:min_len])
 
    print(f"\nTotal tokens evaluated: {len(y_true_all)}")
 
    baseline_f1 = f1_score(y_true_all, y_baseline_all, average="macro", zero_division=0)
    model_f1    = f1_score(y_true_all, y_model_all,    average="macro", zero_division=0)
    improvement = (model_f1 - baseline_f1) / max(baseline_f1, 1e-9) * 100
 
    print("\n" + "="*60)
    print(f"  Baseline F1 (macro):   {baseline_f1:.4f}")
    print(f"  LayoutLMv3 F1 (macro): {model_f1:.4f}")
    print(f"  Improvement:           {improvement:.1f}%")
    print("="*60)
    print("\nDetailed report (LayoutLMv3):")
    print(classification_report(
        y_true_all, y_model_all,
        target_names=LABEL_LIST,
        zero_division=0,
    ))
 
    mlflow.set_tracking_uri("models/mlflow")
    mlflow.set_experiment("document-intelligence")
    with mlflow.start_run(run_name="funsd-evaluation"):
        mlflow.log_metric("baseline_f1_macro", baseline_f1)
        mlflow.log_metric("model_f1_macro", model_f1)
        mlflow.log_metric("f1_improvement_pct", improvement)
        mlflow.log_metric("test_documents", len(examples))
 
    print("\nResults logged to MLflow.")
    return {"baseline_f1": baseline_f1, "model_f1": model_f1, "improvement_pct": improvement}
 
 
if __name__ == "__main__":
    evaluate()
