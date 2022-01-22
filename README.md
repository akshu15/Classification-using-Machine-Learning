# Classification-using-Machine-Learning
The task is to classify whether a patient has diabetes(class 1) or not (class 0), based on the diagnostic measurements provided in the dataset, using logistic regression and neural network as the classifier. The dataset in use is the Pima Indians Diabetes Database(diabetes.csv). The code is written in Python.

# 1.1 Data processing
Extract features values from the data: Process the original CSV data files into a Numpy matrix or Pandas Dataframe. For this we will first import the libraries. We will then use pandas library to load the CSV data to a pandas data frame.

<img width="650" alt="ml1" src="https://user-images.githubusercontent.com/52970601/150614418-2e202a31-5165-4caa-a2ef-698d96cccc48.png">

Data Partitioning:

For this we will first separate the features and target, and then normalize our features.
Using sklearn library’s train-test-split, we will partition our data into training, validation and testing data.
Here we have randomly chosen 60% of the data for training, 20% for validation and the rest for testing.

# 1.2 Implementing Logistic Regression
Train using Logistic Regression: 

We will then define a sigmoid function.

<img width="300" alt="ml2" src="https://user-images.githubusercontent.com/52970601/150614536-bb3623cb-02ce-40ad-81ae-6deeff63379e.png">

A sigmoid 20 function is an activation function with output always lying between a range of 0 to 1.
Now we will define a function for training our model.
In this function we have defined a cost/loss variable where we have used our sigmoid function for calculating the loss and also Gradient Descent for logistic regression to train the model.
Finally we call the model function by passing training set, learning rate and iterations parameters.
Now we will test the performance of our model using the validation set and the testing set. This shows the effectiveness of the model’s generalization power gained by learning.

<img width="467" alt="ml3" src="https://user-images.githubusercontent.com/52970601/150614627-3d81e501-f758-403a-b9fd-d8c4dc36b0a1.png">

# 1.3 Implementing Neural Networks
Train using Neural networks: 

For training the Neural Network model we have used 3 hidden layers with different regularization methods(l2, l1).
As model complexity increases, it is likely that we overfit. One way to control overfitting is adding a regularization term to the error function.
Regularization is used to improve the model’s generalization power gained by learning.
It helps in avoiding overfitting by appending penalties to the loss function. 

L1 Regularization uses the absolute value of the magnitude of coefficient as penalty term to the loss.

<img width="250" alt="ml4" src="https://user-images.githubusercontent.com/52970601/150614729-2bb3cbad-1c56-4e81-8a7e-d18b8c54aa12.png">

where <img width="15" alt="ml_lambda" src="https://user-images.githubusercontent.com/52970601/150614788-e87840d1-fbf5-4319-875e-0c48ba22e6f2.png">
 is the regularization coefficient that controls relative importance of data-dependent error <img width="25" alt="ml_ed" src="https://user-images.githubusercontent.com/52970601/150614799-ed14d64d-9c78-4a07-94d6-3a53daa23419.png"> (w) and regularization term.
After training our model, when we evaluate it we get an accuracy of about 88%.

<img width="467" alt="ml5" src="https://user-images.githubusercontent.com/52970601/150614897-3a15e2b1-a72c-45b7-935e-36774e4b32af.png">

We will then plot the accuracy and loss.

<img width="800" alt="l1_plot" src="https://user-images.githubusercontent.com/52970601/150614950-4d2d53f6-78a1-4f98-8282-0a2a0997ef8a.png">

L2 Regularization uses the squared magnitude of coefficient as the penalty term to the loss.

<img width="250" alt="ml6" src="https://user-images.githubusercontent.com/52970601/150615029-cb394df3-42ad-4e3e-9dff-bbe5c88f09a3.png">

Here after training our model, when we evaluate it we get an accuracy of about 98%.
This is better than the L1 regularization, who shrinks the unimportant feature’s coefficient to zero.
L1 is better in case when we have huge amount of features with us.

<img width="467" alt="l2" src="https://user-images.githubusercontent.com/52970601/150615207-b3fdbfe9-ff5f-4ea2-aabe-697c5244af25.png">

We will then plot the accuracy and the loss for training and valid data.

<img width="800" alt="l2_plot2" src="https://user-images.githubusercontent.com/52970601/150615306-f27d2794-0b32-4db4-890e-8dbd2da7f4a7.png">

In Dropout regularization technique, the neurons are randomly dropped-out. Here I have applied drop out between two hidden layers.
After training our model, when we evaluate it we get an accuracy of about 93%.

<img width="467" alt="dropout" src="https://user-images.githubusercontent.com/52970601/150615355-2ed7d88f-182e-4598-82c8-4dce6b878c4d.png">

We will then plot the accuracy and the loss for training and valid data.

<img width="800" alt="dropout_plot" src="https://user-images.githubusercontent.com/52970601/150615404-88725c57-2fb1-44c5-9f12-df4da7e3f768.png">

For a small number of hidden neurons, we observe that the accuracy of L2 is better than the dropout.
