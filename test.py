import argparse
import os
import torch
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm.auto import tqdm
import numpy as np
from plot import confusionmatrix_plot
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default='data', help="Test Directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size of data")
    parser.add_argument("--model_path", type=str, default=None, help="Path of trained model")
    args = parser.parse_args()
    # Check if resume path exists
    if not os.path.exists(args.model_path):
        print(f"Warning: Model Path '{args.model_path}' does not exist.")
    return args

def model_evalution(model:torch.nn.Module,dataloader:torch.utils.data,loss_fn):
    model.eval()
    target = []
    prediction = []
    total_loss = 0
    total_acc = 0
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            y_pred = model(X)
            loss = loss_fn(y_pred,y)
            total_loss += loss.item()
            target.extend(y)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            prediction.extend(y_pred_class)
            total_acc += (y_pred_class==y).sum().item()/len(y_pred)

    total_acc = total_acc/len(dataloader)
    total_loss = total_loss/len(dataloader)
    print(f"Test Loss: {total_loss} | Train Accuracy: {total_acc}")
    return np.array(target), np.array(prediction)

if __name__=="__main__":
    args = parse_args()
    transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize(size=(48,48), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    test_datasets = datasets.ImageFolder(root=args.test_dir, transform=transform, target_transform=None)
    test_dataloader = DataLoader(dataset=test_datasets,batch_size=args.batch_size,shuffle=False)
    print(f"Total data found: {len(test_datasets)}")
    print(f"Number of batches: {len(test_dataloader)}")
    model = torch.load(args.model_path, weights_only=False)
    print("=========Model load completed==========")
    loss_fn = torch.nn.CrossEntropyLoss() 
    targets, prediction = model_evalution(model,test_dataloader,loss_fn)
    # print(np.unique(prediction))
    # print(np.unique(targets))
    confusionmatrix_plot(targets,prediction,test_datasets.classes)
    
