# Dronetics

Empowering global communities through the convergence of cutting-edge drone technology and advanced AI prediction models to democratize access to critical climate data. We are dedicated to the meticulous collection, analysis, and dissemination of timely and accurate environmental information, focusing on underdeveloped and developed nations. By delivering actionable insights about the imminent environmental threats, particularly droughts and declining air quality, we are committed to fostering resilience and equipping every community, irrespective of economic stature, to effectively confront the challenges posed by a rapidly changing climate. 

Website: https://www.dronetics.org/

Software Architecture Components:

AI Model Implementation

Step 1: 
Define the data pipeline process. The data pipelining process will involve warehousing preprocessed data which will subsequently be cleaned. Then aggregate the normalized data and create a new data directory for feature engineering. 

Step 2: 
Extract the specific features that will be used in the machine learning model from the preprocessed data. Generate additional features based on domain. Combine all the features into one variable. Extract the labels from the preprocessed data. 

Step 3: 
Split the data into a training set and a validation set. Generate a set of models. Add all the trained models into an array. 

Step 4:
For every model in the set of trained models, return a validation score. If the validation score is better than the current best validation score, then update score and model. Return best model. 

Step 5: 
Deploy model to cloud for scalability. Expose model as an API. 

