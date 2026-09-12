"""
prepare_corpus.py — Download AG News and save 5,000 headlines to data/corpus.txt

Run once:
    python prepare_corpus.py
"""
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    output_path = os.path.join("data", "corpus.txt")
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"Corpus already exists at {output_path} with {count} texts. Skipping download.")
        return
    
    print("Downloading AG News dataset from HuggingFace...")
    from datasets import load_dataset
    
    ds = load_dataset("fancyzhx/ag_news", split="train")
    
    # AG News has 4 categories: World, Sports, Business, Sci/Tech
    # Each example has 'text' and 'label'
    # Take 1,250 from each category for balanced representation
    category_names = ["World", "Sports", "Business", "Sci/Tech"]
    per_category = 1250
    
    selected = []
    category_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for example in ds:
        label = example["label"]
        if category_counts[label] < per_category:
            text = example["text"].strip()
            # Clean: take first sentence only if text is very long
            # AG News texts are title + description, we want them short
            if len(text) > 200:
                text = text[:200].rsplit(" ", 1)[0] + "..."
            selected.append(text)
            category_counts[label] += 1
        if all(c >= per_category for c in category_counts.values()):
            break
    
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for text in selected:
            f.write(text + "\n")
    
    print(f"Saved {len(selected)} texts to {output_path}")
    for label, name in enumerate(category_names):
        print(f"  {name}: {category_counts[label]}")


if __name__ == "__main__":
    main()
