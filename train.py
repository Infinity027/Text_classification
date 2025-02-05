import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from plot import plot_image, class_bargraph, acc_plot, loss_plot
from tqdm.auto import tqdm
from model import TextClassificationModelV0
import argparse
import os

class EarlyStopping: #Earlystopping to prevent overfitting
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            if val_loss<self.best_score:
                self.best_score = val_loss
                self.best_model_state = model.state_dict() #save the best model weights
                self.counter = 0
        return False
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data', help="Data Directory Present Train and Test folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size of data")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epoch")
    parser.add_argument("--resume", type=str, default=None, help="Path of trained model")
    args = parser.parse_args()
    # Check if resume path exists
    if args.resume is not None and not os.path.exists(args.resume):
        print(f"Warning: Path '{args.resume}' does not exist. Setting resume to None.")
        args.resume = None  # Set to None if path is invalid
    return args

def training_step(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        b = f"Batch:{batch}/{len(dataloader)}"+"."*(batch%50)
        print(b,end='\r')
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def testing_step(model, dataloader, loss_fn):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            b = f"Batch:{batch}/{len(dataloader)}"+"."*(batch%10)
            print(b,end='\r')
            y_pred = model(X)
            loss = loss_fn(y_pred,y)
            test_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class==y).sum().item()/len(y_pred)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module,train_dataloader,test_dataloader,optimizer,loss_fn: torch.nn.Module,epochs: int = 5, es=None):
  """
     Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
  """
  # 2. Create empty results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}
  
  # 3. Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = training_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer)
    test_loss, test_acc = testing_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn)
    
    # 4. Print out what's happening
    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    # 5. Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    if es!=None and es(test_loss, model):
        print(f"EarlyStopping called model stopped at {epoch+1} epoch")
        es.load_best_model(model)
        break
  # 6. Return the filled results at the end of the epochs
  return results


if __name__=="__main__":
    args = parse_args()
    train_dir = os.path.join(args.data_dir,'train')
    test_dir = os.path.join(args.data_dir,'test')

    transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.Resize(size=(48,48), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    #train dataset load
    train_data = datasets.ImageFolder(root=train_dir,
                                    transform=transform, # a transform for the data
                                    target_transform=None) # a transform for the label/target 
    #test datset load
    test_data = datasets.ImageFolder(root=test_dir,
                                    transform=transform, # a transform for the data
                                    target_transform=None)

    train_dataloader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size=args.batch_size,shuffle=False)
   
    print("Train Data Size:", len(train_data))
    print("Train Data Size:", len(test_data))
    print("Total training batch size:", len(train_dataloader))
    print("Total testing batch size:", len(test_dataloader))
    plot_image(train_data,train_data.classes)
    class_bargraph(train_data, test_data)
    # img, label = next(iter(train_dataloader))
    # print(img.shape,label.shape)
    torch.manual_seed(42) 
    model = TextClassificationModelV0(input_shape=1, output_shape=len(train_data.classes))
    print("Model created.................")
    # Setup loss function and optimizer 
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    #Setup Earlystopping
    es = EarlyStopping(patience=10, delta=0.001)
    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()
    model_results = train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=args.epoch,
                        es=es)

    # End the timer and print out how long it took
    print("------------------------Model run complete--------------------------")
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    # print(type(model_results["test_acc"][0].dtype))
    print("Creating Plot...")
    acc_plot(model_results['train_acc'],model_results['test_acc'])
    loss_plot(model_results['train_loss'],model_results['test_loss'])
    print(f"Model save as {'modelv0_1.pth'}")
    torch.save(model,f='modelv0_1.pth')





     