import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

class Classifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=0.00001):
        super().__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, batch):
        text1_model_inputs, text2_model_inputs, cls_labels = batch
        text1_embedding, _ = self.backbone(**text1_model_inputs, return_dict=False)
        text2_embedding, _ = self.backbone(**text2_model_inputs, return_dict=False)
        text1_embedding = text1_embedding[:,0,:]
        text2_embedding = text2_embedding[:,0,:]
        
        preds = self.log_softmax(torch.mm(text1_embedding, text2_embedding.T))
        targets = torch.LongTensor([i for i in range(preds.shape[0])]).cuda()
        
        loss = self.loss(preds, targets)

        return loss, preds, targets, cls_labels

    def normalize(self, x):
        return x / x.norm(dim=1)[:, None]
    
    def training_step(self, batch, batch_idx):
        loss, preds, targets, _ = self.forward(batch)
        self.log("train_loss", loss.item())

        return {
            "loss": loss
        }

    def training_epoch_end(self, outputs):
        avg_train_loss = np.mean([x["loss"].item() for x in outputs])
        self.log("avg_train_loss", avg_train_loss)

    
    def validation_step(self, batch, batch_idx):
        loss, preds, targets, _ = self.forward(batch)
        self.log("val_loss", loss.item())
        
        return {
            "val_loss": loss.item()
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = np.mean([x["val_loss"] for x in outputs])
        self.log("avg_val_loss", avg_val_loss)
        
        return avg_val_loss
    
    def test_step(self, batch, batch_idx):
        _, preds, _, cls_labels = self.forward(batch)

        return {
            "preds": torch.diagonal(2*torch.sigmoid(preds), 0).detach().cpu().tolist(),
            "cls_labels": cls_labels.detach().cpu().tolist()
        }
        
    def test_epoch_end(self, outputs):
        preds = [pred for x in outputs for pred in x["preds"]]
        cls_labels = [label for x in outputs for label in x["cls_labels"]]
        
        auroc = roc_auc_score(cls_labels, preds)
        self.log("Test AUROC", auroc)
        print(f"AUROC : {auroc}")
        
        
    def configure_optimizers(self):
        grouped_params = [
            {
                "params": list(filter(lambda p: p.requires_grad, self.parameters())),
                "lr": self.learning_rate,
            },
        ]

        optimizer = torch.optim.AdamW(
            grouped_params,
            lr=self.learning_rate, 
        )
        return {"optimizer": optimizer}
