# losses.py
import torch
import torch.nn.functional as F
import torch.jit as jit


class ContrastiveLoss(jit.ScriptModule):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    @jit.script_method
    def forward(self, z_i, z_j, label):
        euclidean_distance = F.pairwise_distance(z_i, z_j)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, margin=1.0, contrastive_weight=1.0, supervised_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(margin)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        self.contrastive_weight = contrastive_weight
        self.supervised_weight = supervised_weight

    def forward(self, latents, labels, preds):
        contrastive_loss = 0
        for i in range(len(latents)):
            for j in range(i + 1, len(latents)):
                label = (labels[i] == labels[j]).float()
                contrastive_loss += self.contrastive_loss(latents[i].unsqueeze(0), latents[j].unsqueeze(0), label)

        contrastive_loss /= (len(latents) * (len(latents) - 1) / 2)
        supervised_loss = self.cross_entropy_loss(preds, labels)

        total_loss = (
            self.contrastive_weight * contrastive_loss + self.supervised_weight * supervised_loss
        )
        return total_loss, contrastive_loss, supervised_loss
