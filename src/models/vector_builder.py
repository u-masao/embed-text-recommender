from sentence_transformers import SentenceTransformer


class VectorBuilder:
    def __init__(self, model_name_or_filepath):
        self.model_name_or_filepath = model_name_or_filepath
        self.model = SentenceTransformer(model_name_or_filepath)

    def encode(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
