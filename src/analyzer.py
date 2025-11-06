# src/analyzer.py
import os
import argparse
import pandas as pd
from textblob import TextBlob

def analyze_text(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def label_from_polarity(p):
    if p > 0.1:
        return "positive"
    elif p < -0.1:
        return "negative"
    else:
        return "neutral"

def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    polarity, subjectivity = analyze_text(text)
    label = label_from_polarity(polarity)
    return {
        "filename": os.path.basename(filepath),
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "label": label
    }

def analyze_folder(folderpath):
    results = []
    for fname in os.listdir(folderpath):
        if fname.lower().endswith(".txt"):
            full = os.path.join(folderpath, fname)
            results.append(analyze_file(full))
    return results

def main():
    parser = argparse.ArgumentParser(description="Simple Sentiment Analyzer (file/folder)")
    parser.add_argument("--file", help="Path to a single .txt file")
    parser.add_argument("--folder", help="Path to a folder containing .txt files")
    parser.add_argument("--out", default="outputs/results.csv", help="CSV output path")
    args = parser.parse_args()

    if not args.file and not args.folder:
        print("Please provide --file or --folder. Example: python src/analyzer.py --file src/data/sample.txt")
        return

    results = [analyze_file(args.file)] if args.file else analyze_folder(args.folder)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Done! Results saved to {args.out}")
    print(df)

if __name__ == "__main__":
    main()
