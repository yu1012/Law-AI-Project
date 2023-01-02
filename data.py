import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data, batch_size=16, max_input_len=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.batch_size = batch_size
        self.max_input_len = max_input_len 
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["train"]+self.data["pair_data"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["val"],
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["test"],
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        text1 = [x[0] for x, _ in batch]
        text2 = [x[1] for x, _ in batch]
        labels = [y for _, y in batch]
                
        text1_model_inputs = self.tokenizer(
            text1,
            max_length=self.max_input_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        text2_model_inputs = self.tokenizer(
            text2,
            max_length=self.max_input_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = torch.Tensor(labels)
        
        return text1_model_inputs, text2_model_inputs, labels