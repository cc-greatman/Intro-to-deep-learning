
import keras
from keras import layers, callbacks
from keras.callbacks import EarlyStopping
import pandas as pd
from ucimlrepo import fetch_ucirepo

#fetch Dataset
ionosphere_filepath = './Datasets/ion.csv'

# data (as pandas dataframes)
ionosphere_data = pd.read_csv(ionosphere_filepath, index_col=0)

# Converting/Remapping the features to binary
ionosphere_data['Class'] = ionosphere_data['Class'].map({'good': 0, 'bad': 1})

df_train = ionosphere_data.sample(frac=0.7, random_state=0)
df_valid = ionosphere_data.drop(df_train.index)

# Scale |0,1|
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Drop the empty feature in Col 2
df_train.dropna(axis=1, inplace=True)
df_valid.dropna(axis=1, inplace=True)

train_x = df_train.drop('Class', axis=1)
valid_x = df_valid.drop('Class', axis=1)
train_y = df_train['Class']
valid_y = df_valid['Class']

rows = train_x.shape[1]

model = keras.Sequential([
    keras.Input(shape=(rows,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['binary_accuracy']
)

early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    train_x, train_y,
    validation_data=(valid_x, valid_y),
    epochs=1000,
    batch_size=512,
    callbacks=[early_stopping],
    verbose=0
)

history_df = pd.DataFrame(history.history)

# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(f"Best Validation Loss: {history_df['val_loss'].min():.4f} +\
      \nBest Validation Accuracy: {history_df['val_binary_accuracy'].min():.4f}")

