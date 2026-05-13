import torch.nn as nn
import torch

class Classification_Module(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.linear_coarse_1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.linear_coarse_2 = nn.Conv1d(embedding_dim, num_classes, kernel_size=1)
        self.linear_fine = nn.Conv1d(embedding_dim, num_classes, kernel_size=1)

        self.dropout = nn.Dropout()

    def forward(self, fine_feats, coarse_feats):
        fine_probs = self.linear_fine(fine_feats)
        fine_probs = fine_probs.permute(0, 2, 1)

        coarse_feats = self.linear_coarse_1(coarse_feats)
        coarse_feats = self.dropout(coarse_feats)
        coarse_probs = self.linear_coarse_2(coarse_feats)
        coarse_probs = coarse_probs.permute(0, 2, 1)

        return fine_probs, coarse_probs
