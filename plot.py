import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_image(data, class_names, figshape=(5,5), seed=None):
    """
    take random image and draw it using plot function
    """
    fig = plt.figure(figsize=(int(figshape[0]*1.8),int(figshape[1]*1.8)))
    if seed:
        np.random.seed(42)
    for i in range(figshape[0]*figshape[1]):
        plt.subplot(figshape[0],figshape[1],i+1)
        rd_idx = np.random.randint(0,len(data))
        img, label = data[rd_idx][0], data[rd_idx][1]
        plt.imshow(img.squeeze(),cmap='gray')
        plt.axis("off")
        plt.title(class_names[label], fontsize=10)
    fig.savefig("imagedraw.png")


def class_bargraph(train_data, test_data):
    """
    draw bar graph of data present in each classes
    """
    class_names = train_data.classes
    data = {    "train": [count for _, count in Counter(train_data.targets).items()],
                "test": [count for _, count in Counter(test_data.targets).items()]
            }
    x = np.arange(len(class_names))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects)
        multiplier += 1
    ax.set_ylabel('count')
    ax.set_title('Number of data present in each classes')
    ax.set_xticks(x + width, class_names)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 1000)

    ax.figure.set_size_inches(25, 10)
    fig.savefig("bargraph.png")

def loss_plot(train_loss, test_loss):
    plt.figure(figsize=(10,7))
    plt.plot(range(len(train_loss)),train_loss,'b', label="Train Loss")
    plt.plot(range(len(test_loss)),test_loss,'g', label="Test Loss")
    plt.legend()
    plt.title("Loss Graph")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("loss_graph.png")

def acc_plot(train_acc, test_acc):
    plt.figure(figsize=(10,7))
    plt.plot(range(len(train_acc)),train_acc,'b',label="Train Accuracy")
    plt.plot(range(len(test_acc)),test_acc,'g', label="Test Accuracy")
    plt.legend()
    plt.title("Accuracy Graph")
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.savefig("acc_graph.png")

def confusionmatrix_plot(target, prediction, num_class:list, save_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(15, 15))
    cm = confusion_matrix(target, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=num_class)
    disp.plot(ax=ax)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    # for _, spine in ax.spines.items():
    #     spine.set_linewidth(0.5)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    