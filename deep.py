import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

concrete_filepath = './Datasets/concrete.csv'

concrete_filedata = pd.read_csv(concrete_filepath)

concrete_filedata.dropna(axis=0)

features = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer',
            'CoarseAggregate', 'FineAggregate', 'Age']

y = concrete_filedata.CompressiveStrength

x = concrete_filedata[features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

input_shape = 8

model = keras.Sequential([
    keras.Input(shape=(input_shape,)),
    # Hidden input layers
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),

    # Linear output layer
    layers.Dense(units=1)
])

# View the total num of rows and cols in a dataframe
# concrete_filedata.shape
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

model.fit(train_x, train_y)

preds = model.predict(val_x)

mae = mean_absolute_error(val_y, preds)

print(f"The mae for a neural network trained model is {mae:.0f}")
