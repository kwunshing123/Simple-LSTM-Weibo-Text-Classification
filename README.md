# Simple LSTM Weibo Text Classification

This a simple text classification project through LSTM. The model used Keras framework in Python to train. This project will simply included two parts - Backend and Front-end.

The Backend folder included the trained model and the API server.

The Frontend folder included a webpage to call the api.

## Dataset

- This dataset is downloaded from the Internet.

- The path of dataset will be:  
```/Backend/data/```

- The data included three categories that are design, health, and technology.

## Train the model

- Running ```python keras_lstm.py``` to train the model

- The architecture will be Word Segmentation -> Embedding -> LSTM

- This model seem over fitting

- This trained model saved as:  
```/Backend/keras_lstm.h5```

## Backend

- The Backend server will be using Flask

- The port of the API server will be ```5002```

- For running the server:
```python server.py```

## Front-End

- The Front-End will be written by simple Node.js throguh the Express framework

- The port of the webpage will be ```8080```

- For host the web server:
```node app.js```
