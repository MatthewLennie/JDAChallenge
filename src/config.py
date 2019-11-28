


#Test Train Split
training_data_url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
x_keys = ['mnth', 'hr', 'holiday', 'weekday',
               'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
y_keys = 'cnt'
random_seed = 42

#HyperParameter Search 
parameters = {'reg_lambda':[0.1],'reg_alpha':[0.1],'n_estimators':[100],'gamma':[0.9],'max_depth':[10,11],'learning_rate':[0.5]}
n_jobs = -1
number_of_cross_validations =3