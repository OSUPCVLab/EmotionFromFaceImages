## IMPORTS

import os, argparse

from model import MotivDataModule, MotivNet

import lightning as L

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Motiv dataset")
    
    parser.add_argument("--train_data", type=str, help="Path to the folder containing the training images")
    parser.add_argument("--train_labels", type=str, help="Path to the training labels")

    parser.add_argument("--test_labels", type=str, help="Path to the test labels")
    parser.add_argument("--test_data", type=str, help="Path to the folder containing the test images")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--encoder_lr", type=float, default=1e-8, help="Learning rate for training")
    parser.add_argument("--decoder_lr", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes in the dataset")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use for training")
    parser.add_argument("--train_test_split", type=float, default=0.8, help="Proportion of data to use for training")
    args = parser.parse_args()

    if not all([args.train_data, args.train_labels, args.test_data, args.test_labels]):
        raise ValueError("Please provide all required data paths.")
        exit(1)

    # need 4 params, folder for traing data images, json file with train labels, folder for test data images, json file with test labels

    data_module = MotivDataModule(
        train_data=args.train_data,
        train_labels=args.train_labels,
        test_data=args.test_data,
        test_labels=args.test_labels,
        train_test_split=args.train_test_split,
        batch_size=args.batch_size
    )

    data_module.prepare_data()
    data_module.setup(stage="fit")

    model = MotivNet(args.num_classes, args.encoder_lr, args.decoder_lr)

    logger = CSVLogger("logs/", name="motivnet")

    trainer = L.Trainer(
        devices=args.gpus,
        max_epochs=args.epochs,
        accelerator="cuda" if args.gpus > 0 else "cpu",
        logger=logger,
        log_every_n_steps=0,
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                mode="min",
                save_on_train_epoch_end=True,
                monitor="val_loss",
                save_last=True
            )
        ]
    )

    trainer.fit(model, data_module)

    data_module.setup(stage="test")

    trainer.test(model, data_module.test_dataloader())