import gc
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MLEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.search_model = None
        self.nli_model = None
        self.rerank_model = None
        self.tokenizer = None
        self.id2label = None

        self.search_model_path = '../hugg'
        self.rerank_model_path = '../rerank'
        self.nli_model_path = '../bert'

    def load_models(self):
        print("Loading models ")
        try:
            self.search_model = SentenceTransformer(self.search_model_path)
            self.search_model.to(self.device)

            self.rerank_model = CrossEncoder(self.rerank_model_path, device=self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_path)

            self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_path)
            self.nli_model.to(self.device)

            self.id2label = self.nli_model.config.id2label
        except Exception as e:
            print(f"Error loading models: {e}")
            raise e

    def upload_models(self):
        self.search_model = None
        self.nli_model = None
        self.tokenizer = None
        self.id2label = None

        gc.collect()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()


ml_engine = MLEngine()