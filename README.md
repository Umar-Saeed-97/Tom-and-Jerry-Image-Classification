# Tom & Jerry Image Classification

Tom and Jerry Image Classification is a computer vision project that classifies frames from the classic animated show, Tom and Jerry, into one of four categories: Just Tom, Just Jerry, Both Tom and Jerry, and Neither Tom nor Jerry. This project uses state-of-the-art deep learning techniques to accurately identify the presence of the two iconic characters in each episode frame. The ultimate goal of this project is to gain a better understanding of the distribution of the characters throughout the series and provide an entertaining and informative tool for fans of the show. Whether you're a casual viewer or a die-hard fan, this project promises to add an exciting new dimension to your appreciation of one of the most beloved animated series of all time.


## Requirements

The following packages are required to run the notebook:

- matplotlib
- numpy
- tensorflow
- pandas
- splitfolders
- opencv-python
- seaborn
- scikit-learn
- squarify

You can install these packages using the following pip command:

```bash
  pip install matplotlib numpy tensorflow pandas splitfolders opencv-python seaborn scikit-learn squarify
```
## Usage

Here's how you can use the code in this project:

1. Clone or download the repository to your local machine.
 
2. Download the dataset from Kaggle (link provided below).

3. Open the Jupyter Notebook file in the repository.

4. Run the cells in the notebook to train the model and make predictions.

5. You can modify the code to fit your needs, such as changing the model architecture, training parameters, and prediction logic.

Note: Make sure to activate the virtual environment each time you work on the project to have the required packages installed.

## Dataset

The dataset used in this project is the Tom and Jerry Image Classification dataset and can be obtained from [Kaggle](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification).

The dataset contains 5478 images and is divided into 4 categories: "Just Tom", "Just Jerry", "Both Tom and Jerry", and "Neither Tom nor Jerry".

Note: Please review and abide by the terms of use and license information for the dataset before using it in your own projects.
## Model


The model used in this project is based on EfficientNetV2B0, with additional layers added for the Tom and Jerry classification task. The model architecture is as follows:

- An input layer with a shape of (224, 224, 3)
- A data augmentation preprocessing layer
- The EfficientNetV2B0 model, with the top layer removed, as a base model
- A dropout layer with a rate of 0.2 to prevent overfitting
- A GlobalAveragePooling2D layer
- Another dropout layer with a rate of 0.3
- A dense layer with 4 neurons
- An activation layer with the activation function set to softmax

The model was compiled with categorical crossentropy loss, Adam optimizer, and accuracy metric. The entire base model layers were unfrozen for fine-tuning.
## Training Process

The model was trained for 20 epochs using the `fit` method from the Keras API. The training data was passed to the method as an argument, along with the number of steps per epoch and the validation data.

Three callbacks were added to monitor the training process: model checkpoint, learning rate reduction, and a CSV logger.

After the initial training phase with the layers frozen, the model was fine-tuned by unfreezing all layers of the base model. The layers were set to be trainable and the model was trained again using the same process as before. 

## Evaluation

The model was evaluated on a test dataset which was 10% of the original dataset. The model achieved an accuracy score of 93.08% on the test dataset. A classification report and a confusion matrix were also generated to provide a more comprehensive understanding of the model's performance. The classification report displayed the precision, recall and f1-score for each of the 4 categories. The confusion matrix provided a graphical representation of the number of correct and incorrect predictions made by the model. Both the classification report and the confusion matrix were plotted to visualize the results and to gain further insights into the model's performance.
## Future Work

- Training the model with more data or different datasets to improve the accuracy score.
- Experimenting with other image classification models such as ResNet, InceptionV3, and Xception to compare their performance.
- Incorporating additional layers to enhance the model's performance. 

## License

MIT


## Author

[Umar Saeed](https://www.linkedin.com/in/umar-saeed-16863a21b/)

