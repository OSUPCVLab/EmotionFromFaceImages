## IMPORTS
import os, argparse, cv2

from model import MotivNet
from torch.utils.data import DataLoader, Dataset
import lightning as L

import torch

class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

        self.transform_mean=torch.tensor([123.5, 116.5, 103.5])
        self.transform_std=torch.tensor([58.5, 57.0, 57.5])    
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, (768,1024), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image[[2, 1, 0], ...].float()
        m = self.transform_mean.view(-1, 1, 1)
        s = self.transform_std.view(-1, 1, 1)
        image = (image - m) / s
        return self.image_paths[idx], image

if __name__ == "__main__":
    # parse cli arguments
    parser = argparse.ArgumentParser(
        prog="MotivNet",
        description="Enhancing Meta-Sapiens as a emotionally intelligent foundational model",
        epilog="Inference script for MotivNet",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to folder containing inference images",
    )

    parser.add_argument(
        "--GPUS",
        type=int,
        default=1,
        help="Number of GPUs to use for inference",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of top predictions to return",
    )

    args = parser.parse_args()

    if args.input == "":
        raise ValueError("Please provide a path to the inference data")

    if not os.path.exists(args.input):
        raise FileNotFoundError("The provided path does not exist")

    # load data

    print("Loading data...")
    print("Input path:", args.input)
    print(str(len(os.listdir(args.input))) + " file(s) found")

    dataset = ImageDataset(args.input)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # load model

    print("Loading model...")

    model = MotivNet(8,0,0)
    modelload_state_dict(torch.load(os.path.join(os.getcwd(), "checkpoints", "MotivNet.pth"), weights_only=True))
    trainer = L.Trainer(devices=args.GPUS)

    # inference

    print("Running inference...")

    preds = trainer.predict(model, dataloader)

    print("Predictions:")

    file_names, preds = preds[0]

    results = []
    for file_name, pred in zip(file_names, preds):
        top_k_indices = torch.topk(pred, args.top_k).indices.tolist()
        results.append((file_name, top_k_indices))

    print(results)

