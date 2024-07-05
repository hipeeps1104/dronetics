# Dronetics

Empowering global communities through the convergence of cutting-edge drone technology and advanced AI prediction models to democratize access to critical climate data. We are dedicated to the meticulous collection, analysis, and dissemination of timely and accurate environmental information, focusing on underdeveloped and developed nations. By delivering actionable insights about the imminent environmental threats, particularly droughts and declining air quality, we are committed to fostering resilience and equipping every community, irrespective of economic stature, to effectively confront the challenges posed by a rapidly changing climate. 

Website: https://www.dronetics.org/

Software Architecture Components:

AI Model Implementation

1. Data Collection and Processing: Define the data pipeline process. Data pipelining process will involve warehousing preprocessed data which will be cleaned. Scale to a standard range to help improve the performance of the algorithm. Use statistical methods to fill missing data values in order to reduce bias in data set. Aggregate the normalized data and create a new data directory for feature engineering.
2. Feature Engineering: Extract the specific features that will be used in the machine learning model from the preprocessed data. Generate domain-specific features to help reduce feature mismatch between the source and target domains. Combine all the features into one data set. Extract the labels from the preprocessed data.
3. Model Training. Split the data into a training set and a validation set. Generate a set of models such as: Random Forest, Gradient Boosting, Neural Network. Train the models with the training set. Add all the trained models into a data set.
4. Model Validation. For every model in the set of trained models, return a validation score. If the validation score is better than the current best validation score, then update score and model. Return best model.
5. Model Depolyment. Deploy model to cloud for scalability. Expose model as an API. 


