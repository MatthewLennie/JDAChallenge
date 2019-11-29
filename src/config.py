

# Test Train Split
training_data_url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
x_keys = ['mnth', 'hr', 'holiday', 'weekday',
          'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
y_keys = 'cnt'
random_seed = 42

# HyperParameter Search
parameters = {
    'reg_lambda': [
        0.1, 0.3, 0.5, 1], 'reg_alpha': [
            0.1, 0.3, 1], 'n_estimators': [
                100, 200], 'gamma': [
                    0.1, 0.2, 0.9, 0.01], 'max_depth': [
                        2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14], 'learning_rate': [
                            0.01, 0.1, 0.3, 0.5, 0.9]}

n_jobs = -1
number_of_cross_validations = 3
