import gc
import torch
import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MLEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.search_model = None
        self.rerank_model = None  # Добавляем для Cross-Encoder
        self.nli_model = None
        self.tokenizer = None
        self.id2label = None
        self.bm25 = None  # Добавляем для BM25
        
        # Пути к твоим моделям[cite: 1, 8]
        self.search_model_path = '../hugg'
        self.rerank_model_path = '../rerank' 
        self.nli_model_path = '../bert'
        self.bm25_path = 'bm25_index.pkl'

    def load_models(self):
        print("--- Загрузка продвинутых ML-моделей ---")
        try:
            # 1. Поисковая модель (Bi-Encoder)
            self.search_model = SentenceTransformer(self.search_model_path, device=self.device)
            
            # 2. Модель переранжирования (Cross-Encoder)
            if os.path.exists(self.rerank_model_path):
                self.rerank_model = CrossEncoder(self.rerank_model_path, device=self.device)
                print("Cross-Encoder загружен успешно")
            
            # 3. NLI модель для верификации[cite: 1]
            self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_path)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_path).to(self.device)
            self.id2label = self.nli_model.config.id2label
            
            # 4. Загрузка индекса BM25 (если он уже создан миграцией)[cite: 8]
            if os.path.exists(self.bm25_path):
                with open(self.bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                print("Индекс BM25 загружен")
                
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            raise e

    def upload_models(self):
        self.search_model = None
        self.rerank_model = None
        self.nli_model = None
        self.bm25 = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

ml_engine = MLEngine()