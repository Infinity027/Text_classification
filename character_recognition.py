import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import datasets
import os
from PIL import Image
from Segment import CharacterSegmentation

class CharacterDataset(Dataset):
    def __init__(self, characters, transform=None):
        """
        characters: List of segmented character images (numpy arrays).
        transform: Transformations to apply to each image.
        """
        self.characters = characters
        self.transform = transform

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        # Convert OpenCV (numpy) image to PIL format
        char_pil = Image.fromarray(self.characters[idx])

        # Apply transformations if available
        if self.transform:
            char_pil = self.transform(char_pil)

        return char_pil  # Return image tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='data', help="Text image path")
    parser.add_argument("--c", type=int, default=4, help="dilate mode iteration")
    parser.add_argument("--model_path", type=str, default=None, help="Path of trained model")
    parser.add_argument("--draw_plot", type=bool, default=False, help="Draw plot of Images of characters")
    args = parser.parse_args()
    # Check if resume path exists
    if not os.path.exists(args.image_path):
        print(f"Warning: Image Path '{args.model_path}' does not exist.")
        exit(1)
    if not os.path.exists(args.model_path):
        print(f"Warning: Model Path '{args.model_path}' does not exist.")
        exit(1)
    
    return args

if __name__=="__main__":
    args = parse_args()
    cs = CharacterSegmentation(draw_plot=args.draw_plot)
    characters, spaces = cs.get_img(image_path=args.image_path,c=args.c)
    print("Total characters found:",len(characters))
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(48,48), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    model = torch.load(args.model_path, weights_only=False)
    char_dataset = CharacterDataset(characters, transform=transform)
    char_loader = DataLoader(char_dataset, batch_size=32, shuffle=False)
    model.eval()  
    predictions = []
    #load test dataset to get idx to class conversion
    test_datasets = datasets.ImageFolder(root='data/test', target_transform=None)
    idx_to_class = {v: k for k, v in test_datasets.class_to_idx.items()}

    with torch.no_grad():
        for batch in char_loader:
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1).numpy()  # Get predictions
            predictions.extend(preds)

    sentence = ''
    for i,pred in enumerate(predictions):
        sentence += idx_to_class[pred]
        if (i+1) in spaces:
            sentence += ' '
    print(f"Recognize test: '{sentence}'")
    f = open("recognize.txt", "w")
    f.write(sentence)
    f.close()

    
