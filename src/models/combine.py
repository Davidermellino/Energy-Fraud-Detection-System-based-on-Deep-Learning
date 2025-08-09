import numpy as np


class Combiner:
    def __init__(self, embeddings_path, num_ids=100, embedding_dim=17, num_datasets=5):
        self.embeddings_path = embeddings_path
        self.embeddings_list = self.load_embeddings()
        self.num_ids = num_ids
        self.embedding_dim = embedding_dim
        self.num_datasets = num_datasets

    def load_embeddings(self):
        """Load all embeddings once and return as list"""
        embeddings_list = []
        for file in self.embeddings_path.glob("*.npy"):
            print(f"Loading embeddings from {file.name}")
            embeddings = np.load(file)
            embeddings_list.append(embeddings)
        return embeddings_list

    def concatenate_embeddings(self):
        """Concatenate embeddings by id"""

        # Get all unique IDs from the first embedding file to determine the actual range
        first_embedding = self.embeddings_list[0]
        unique_ids = np.unique(first_embedding[:, 0].astype(int))
        actual_num_ids = len(unique_ids)

        combined_embeddings = np.zeros(
            (actual_num_ids, self.embedding_dim * self.num_datasets)
        )

        for idx, supply_id in enumerate(unique_ids):
            embedding_for_id = []
            for embedding in self.embeddings_list:
                mask = embedding[:, 0].astype(int) == supply_id

                embedding_with_id = embedding[mask]

                if embedding_with_id.size > 0:
                    embedding_with_id = embedding_with_id[0, 1:]
                    embedding_for_id.append(embedding_with_id)
                else:
                    embedding_for_id.append(np.zeros(self.embedding_dim))

            concatenated_embedding = np.concatenate(embedding_for_id, axis=0)
            combined_embeddings[idx, :] = concatenated_embedding

        return combined_embeddings
