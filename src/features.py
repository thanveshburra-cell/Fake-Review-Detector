import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class StylometricExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        meta_features = []
        for text in X:
            text_str = str(text)
            words = text_str.split()
            char_len = len(text_str)
            word_len = len(words)
            lexical_diversity = len(set(words)) / (word_len + 1e-6)
            avg_word_len = char_len / (word_len + 1e-6)
            exclamations = text_str.count('!') / (word_len + 1e-6)
            questions = text_str.count('?') / (word_len + 1e-6)
            caps_ratio = sum(1 for c in text_str if c.isupper()) / (char_len + 1e-6)
            meta_features.append([
                char_len, word_len, avg_word_len, lexical_diversity,
                exclamations, questions, caps_ratio
            ])
        return np.nan_to_num(np.array(meta_features))

class FeatureEngineeringPipeline:
    def __init__(self):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=8000, sublinear_tf=True)
        self.cat_encoder = OneHotEncoder(handle_unknown='ignore')
        self.meta_extractor = StylometricExtractor()
        self.scaler = StandardScaler()

    def fit(self, raw_texts, normalized_texts, categories):
        self.word_vec.fit(normalized_texts)
        self.char_vec.fit(normalized_texts)
        self.cat_encoder.fit(np.array(categories).reshape(-1, 1))
        meta = self.meta_extractor.fit_transform(raw_texts)
        self.scaler.fit(meta)
        return self

    def transform(self, raw_texts, normalized_texts, categories):
        w_feat = self.word_vec.transform(normalized_texts)
        c_feat = self.char_vec.transform(normalized_texts)
        cat_feat = self.cat_encoder.transform(np.array(categories).reshape(-1, 1))
        m_feat = self.scaler.transform(self.meta_extractor.transform(raw_texts))
        return hstack([w_feat, c_feat, cat_feat, csr_matrix(m_feat)]).tocsr()