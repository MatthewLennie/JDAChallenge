#Configuration file for the XGBoost implementation for the Biekshare dataset. 

# Test Train Split
training_data_url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
x_keys = ['mnth', 'hr', 'holiday', 'weekday',
          'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
y_keys = 'cnt'
random_seed = 42

# HyperParameter Search
# I ran a HyperParameter Search already and have constrained the search. 
parameters = {
    'reg_lambda': [
         0.2, 0.3, 0.4], 'reg_alpha': [
            0, 0.05,0.1,0.2, 0.3], 'n_estimators': [
                200,300], 'gamma': [
                    0.1, 0.2, 0.01], 'max_depth': [
                        4, 6, 7, 8], 'learning_rate': [
                            0.01, 0.09, 0.1, 0.11]}

n_jobs = -1
number_of_cross_validations = 3
