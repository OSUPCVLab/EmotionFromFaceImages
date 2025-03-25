##### IMPORTS ######
import os, shutil, json, re, time, cv2, pickle, argparse
import numpy as np

# create new folder called temp if the os tmpdir is not big enough to store checkpoints
# os.environ["TMPDIR"] = os.getcwd() + "/temp"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import torchmetrics as tm

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from ml_decoder import MLDecoder

##### CONSTANTS ######

SAPIENS_POSE_TRANSFORMER = os.path.join(os.getcwd(), "checkpoints", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")
INPUT_CHANNELS = 1536

##### FUNCTIONS ######

# map image files maps to List[np.ndarray]
def collate_fn(batch, data_path):
    output_x, output_y = [], []

    mean=torch.tensor([123.5, 116.5, 103.5])
    std=torch.tensor([58.5, 57.0, 57.5])

    for file_path, label in batch:
        image = cv2.imread(os.path.join(data_path, file_path))
        image = cv2.resize(image, (768,1024), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image[[2, 1, 0], ...].float()
        m = mean.view(-1, 1, 1)
        s = std.view(-1, 1, 1)
        image = (image - m) / s
        output_x.append(torch.tensor(image, dtype=torch.float32))
        output_y.append(torch.tensor(label, dtype=torch.int64))

    return torch.stack(output_x), torch.stack(output_y)

##### CLASSES ######

class MotivDataSet(Dataset):
    def __init__(self, data):
        # data is a list of tuples [(9.jpg, 1), ...]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
      
class MotivDataModule(L.LightningDataModule):
    def __init__(self, train_data: str, train_labels: str, test_data:str, test_labels: str, train_test_split: float, batch_size: int
    ):
        super().__init__()
        self.training_data = []
        self.test_data = []

        self.train_labels = os.path.join(os.getcwd(), train_labels)
        self.test_labels = os.path.join(os.getcwd(),test_labels)

        self.train_data_path = os.path.join(os.getcwd(), train_data)
        self.test_data_path = os.path.join(os.getcwd(), test_data)

        self.train_test_split = train_test_split
        self.batch_size = batch_size
    
    def prepare_data(self):

        with open(self.train_labels, 'r') as f:
            label_data = json.load(f)
            for emotion_label in label_data:
                for data_title in label_data[emotion_label]:
                    self.training_data.append((data_title, int(emotion_label)))
        
        with open(self.test_labels, 'r') as f:
            label_data = json.load(f)
            for emotion_label in label_data:
                for data_title in label_data[emotion_label]:
                    self.test_data.append((data_title, int(emotion_label)))
    
    def setup(self, stage):
        match stage:
            case "fit":
                dataset = MotivDataSet(self.training_data)
                self.train_set, self.validation_set = random_split(dataset, [int(self.train_test_split * len(dataset)), len(dataset) - int(self.train_test_split * len(dataset))])
            case "test":
                self.test_set = MotivDataSet(self.test_data)
            case _:
                pass
    
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size, collate_fn=lambda batch: collate_fn(batch, self.train_data_path))
    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, collate_fn=lambda batch: collate_fn(batch, self.train_data_path))
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=lambda batch: collate_fn(batch, self.test_data_path))

class MotivNet(L.LightningModule):
    def __init__(self, num_classes:int, encoder_lr:float, decoder_lr:float):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        
        pose_transformer = torch.jit.load(SAPIENS_POSE_TRANSFORMER)
        self.pose_encoder = pose_transformer.backbone.train()
        for param in self.pose_encoder.parameters():
            param.requires_grad = True

        self.decoder = MLDecoder(num_classes=num_classes, initial_num_features=INPUT_CHANNELS)

        self.val_auroc = tm.classification.MulticlassAUROC(num_classes=num_classes)
        self.test_auroc = tm.classification.MulticlassAUROC(num_classes=num_classes)
        self.val_accuracy = tm.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_accuracy = tm.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = tm.classification.Accuracy(task="multiclass", num_classes=num_classes)
    
    def training_step(self, batch, batch_idx):
        X, y = batch

        r = self.pose_encoder(X)
        r = self.decoder(r)

        loss = nn.CrossEntropyLoss()(r, y)
        self.train_accuracy(r, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch

        r = self.pose_encoder(X)
        r = self.decoder(r)

        loss = nn.CrossEntropyLoss()(r, y)
        self.val_accuracy(r, y)
        self.val_auroc(r, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch

        r = self.pose_encoder(X)
        r = self.decoder(r)

        loss = nn.CrossEntropyLoss()(r, y)
        self.test_accuracy(r, y)
        self.test_auroc(r, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_auroc", self.test_auroc, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch):
        file_path, image = batch
        r = self.pose_encoder(image)
        r = self.decoder(r)

        preds = nn.Softmax(dim=1)(r)

        return file_path, preds

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {"params": self.pose_encoder.parameters(), "lr": self.encoder_lr},
            {"params": self.decoder.parameters(), "lr": self.decoder_lr}
        ])
        scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1),
            "monitor": "val_loss"
        }

        return [optimizer], [scheduler]