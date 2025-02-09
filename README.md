# <div align="center">Character Classification</div>
<p align="center">
  <img alt="Characters" src="imagedraw.png" title="Characters" width="50%"/>
</p>

---

## [Content]
1. [Description](#description)   
2. [Installation](#installation)  
3. [Model Training](#model-training)
4. [Model Testing](#model-testing)
5. [Text Segmentation](#text-segmentation)

---
## [Description]

This project is a deep learning-based character classification model that recognizes 80 different types of characters using convolutional neural networks (CNNs). The model is trained on a dataset of characters and can predict the class of a given character image with high accuracy.
![Available Data for all classes(Training & Testing)](bargraph.png 'Available Data for all classes(Training & Testing)')

## [Installation]
1.Clone the Repository
```python
git clone https://github.com/Infinity027/Text_classification.git
cd Text_classification
```
2. Install Dependencies
```python
pip install -r requirements.txt
```

## [Model Training]
To train the model on your dataset, run:
```python
python3 train.py --data_dir "data" --batch_size 32 --epoch 100
```

Model Performance:
1. Training data Accuracy: 96.43%
2. Testing data Accuracy: 96%


<img src="acc_graph.png" title="Loss Graph" align="left" width="50%"> <img src="loss_graph.png" title="Accuracy Graph" align="right" width="50%">
<br/>

## [Model Testing]
To test the model run the test model, it will generate confusion matrix of testing data:
```python
python3 testpy --test_dir "data/test" --batch_size 32 --model_path 'Modelv0_1.pth'
```
![Confusion Matrix](confusion_matrix.png)

## [Text Segmentaion]
for character segmentation from text images, I used OpenCV. It involves preprocessing images, detecting text lines, segmenting words and characters, and using a trained deep learning model for recognition. run following code for character segmentation
```python
python3 Segment.py --image_path "test/test.png" --c 4 --min_area 100 --draw_plot 0
```
![Character Segmentation](result/line_1.png)

Same 'character_recognition.py' python file extract the characters images from text image then sent it to model and predict the characters
```python
python3 character_recognition.py --image_path "test/test.png" --c 4 --model_path 'modelv0_1.pth' --draw_plot 0
```
output: 'It was the best of times, it was the worst of times, it was tbe age of wisdom, it was the agc of foolishness '
