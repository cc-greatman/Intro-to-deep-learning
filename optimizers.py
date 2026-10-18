import keras
from keras import layers
import pandas as pd
from IPython.display import display

concrete_filepath = './Datasets/concrete.csv'

concrete_filedata = pd.read_csv(concrete_filepath)

concrete_filedata.dropna(axis=0)

# Create training and validation data
df_train = concrete_filedata.sample(frac=0.7, random_state=0)
df_valid = concrete_filedata.drop(df_train.index)

display(df_train.head())

# Scale to |0,1|
min_ = df_train.min(axis=0)
max_ = df_train.max(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and targets
train_x = df_train.drop('CompressiveStrength', axis=1)
valid_x = df_valid.drop('CompressiveStrength', axis=1)
train_y = df_train['CompressiveStrength']
valid_y = df_valid['CompressiveStrength']

cols, rows = train_x.shape

# Definie the model
model = keras.Sequential([
    keras.Input(shape=(rows,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1)
])

# Compile the model with an optimizer and loss function
model.compile(
    optimizer='adam',
    loss='mae'
)

# Train the model
history = model.fit(
    train_x, train_y,
    validation_data=(valid_x, valid_y),
    batch_size=256,
    epochs=10
)

# Using a graph to plot the change in loss
# Converting the history to a Dataframe
history_df = pd.DataFrame(history.history)

# Use panda's native plot method
history_df['loss'].plot()