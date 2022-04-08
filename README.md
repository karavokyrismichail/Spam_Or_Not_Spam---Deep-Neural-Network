# Spam_Or_Not_Spam---Deep-Neural-Network

Projet for the **Data Mining and Machine Learning** course of the [Department of Computer Engineering & Informatics](https://www.ceid.upatras.gr/en).

## Desription 

I try to predict if a given email is spam or not.
<br>
<br> The [dataset](https://github.com/karavokyrismichail/Spam_Or_Not_Spam---Deep-Neural-Network/tree/main/spam_data) has two columns. The first column contains the emails and the second column contains 1 if the email in that row is spam and 0 if the email in that row is not spam. 
<br>
First, I turn the texts into vectors, using the [TF-IDF vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), so I can put them as inputs to the neural network model. Then I split the dataset to 75% train and 25% test. After, I create a neural network model with one hidden layer. For the output layer, I use the sigmoid activation function for binary classification. Finally I measure the performance of the DNN model.


## Results

|  | **precision_score** | **f1_score** | **recall_score** |
|--|--|--|--|
| Epochs = 4 | 1 | 0.9874476987 | 0.958677686 |
| Epochs = 6 | 1 | 0.9874476987 | 0.9752066116 |
| Epochs = 8 | 1 | 0.9874476987 | 0.9752066116 |
