import torch


class PrecomputedEmbeddingsDataset(torch.utils.data.Dataset):
    """
    Custom wrapper dataset to store precomputed sentence or audio embeddings,
    labels, and preserve required metadata/attributes like LABELS_EMOTIONS.
    """

    def __init__(self, embeddings, labels, original_dataset):
        self.embeddings = embeddings
        self.labels = labels
        self.LABELS_EMOTIONS = original_dataset.LABELS_EMOTIONS

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
