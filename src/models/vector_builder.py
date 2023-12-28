import numpy as np
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer


class VectorBuilder:
    def __init__(
        self, model_name_or_filepath, chunk_overlap=50, tokens_par_chunk=None
    ):
        self.model_name_or_filepath = model_name_or_filepath
        self.model = SentenceTransformer(model_name_or_filepath)
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            model_name=model_name_or_filepath,
            tokens_per_chunk=tokens_par_chunk,
        )

    def encode(self, sentences):
        embeddings = []
        for sentence in sentences:
            vectors = self.model.encode(self.splitter.split_text(sentence))
            mean_vector = np.mean(vectors, axis=0)
            assert (
                len(mean_vector)
                == self.model.get_sentence_embedding_dimension()
            )
            embeddings.append(mean_vector)
        result = np.array(embeddings)
        assert result.shape[0] == len(sentences)
        assert result.shape[1] == self.model.get_sentence_embedding_dimension()
        return result
