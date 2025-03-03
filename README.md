# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1. Preprocess the dataset using feature scaling.
   
2. Train a neural network to predict outputs based on given inputs.
   
3. Minimize error using the RMSprop optimizer and the MSE loss function.
   
4. Evaluate model performance on test data.
   
5. Visualize the training loss over epochs.
   
6. Make predictions using the trained model.


## Neural Network Model

![image](https://github.com/user-attachments/assets/ee9acc10-42da-48f5-9a05-b860601c1f28)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:LEANN JOBY MATHEW
### Register Number:212222230074
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        class NeuralNet(nn.Module):
          self.fc1 = nn. Linear (1, 4)
          self.fc2 = nn. Linear (4, 2)
          self.fc3 = nn. Linear (2, 1)
          self.relu = nn. ReLU()
          self.history = {'loss': []}
  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self. fc3(x)
    return x
```

```python
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=4000) :
  for epoch in range (epochs) :
    optimizer. zero_grad()
    loss = criterion(ai_brain(X_train), y_train)
    loss. backward()
    optimizer.step()
    ai_brain. history['loss'] .append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information

![image](https://github.com/user-attachments/assets/b560342c-3a35-47ad-812f-29808c6959ab)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/101c949a-d33b-4780-be31-41634e6fb8e3)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/ffdc1b0d-d1cc-4199-9226-7ca81d4c28b8)


## RESULT
Thus a neural network regression model is developed successfully.The model demonstrated strong predictive performance on unseen data, with a low error rate.
