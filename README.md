# widebot-task2
The code of the second assignment of widebot internship

-> The Final version of the code [here](https://github.com/moaaztaha/widebot-task2/blob/master/Final.ipynb)

-> Description of the steps included in the jupyter notebook file

- `predict.py`
	- Loads the **preprocessing pipeline** for the features
	- Loads the **label encoder** for the target column
	- Loads the best performing models
		- `knn_train.pkl`: best performing on the training data
		- `rf_all.pkl`: best performing on all the data
	- Loads one random row from the validation set 
	- Preprocess the row
	- Make prediction

- Docker
	- `docker build -t binary . && docker run -it binary`
