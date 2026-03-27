from pathlib import Path
import json
 
OUTPUT_DIR = Path("data/annotations/funsd")
 
 
def download_funsd():
    from datasets import load_dataset
 
    print("[prepare_funsd] Downloading FUNSD from HuggingFace...")
    ds = load_dataset("nielsr/funsd-layoutlmv3")
 
    sample = ds["train"][0]
    print(f"[prepare_funsd] Available fields: {list(sample.keys())}")
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    for split in ("train", "test"):
        records = []
        for example in ds[split]:
            words  = example.get("words") or example.get("tokens") or []
            bboxes = example.get("bboxes") or example.get("bbox") or []
            labels = (
                example.get("ner_tags")
                or example.get("labels")
                or example.get("label")
                or []
            )
            records.append({
                "id":         example.get("id", ""),
                "words":      words,
                "bboxes":     bboxes,
                "labels":     labels,
                "image_path": example.get("image_path", ""),
            })
 
        out_path = OUTPUT_DIR / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[prepare_funsd] Saved {len(records)} examples → {out_path}")
 
    print("[prepare_funsd] Done.")
    print(f"  Train: {len(ds['train'])} documents")
    print(f"  Test:  {len(ds['test'])} documents")
 
 
if __name__ == "__main__":
    download_funsd()
