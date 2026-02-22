from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerWrapper:
    def __init__(self):
        # small, safe defaults; you can tune later
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
        )
        self._fitted = False

    def fit_reference_csv(self, csv_path: Path):
        """
        Loads reference CSV and fits TF-IDF on the reference texts.
        Expected columns: item_id, text
        Returns: (ref_df, ref_matrix)
        """
        df = pd.read_csv(csv_path)

        required = {"item_id", "text"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Reference CSV missing columns: {sorted(missing)}")

        df = df.copy()
        df["item_id"] = df["item_id"].astype(str)
        df["text"] = df["text"].fillna("").astype(str)

        X = self.vectorizer.fit_transform(df["text"].tolist())
        self._fitted = True
        return df, X

    def transform_query(self, text: str):
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit_reference_csv first.")
        if text is None:
            text = ""
        return self.vectorizer.transform([str(text)])