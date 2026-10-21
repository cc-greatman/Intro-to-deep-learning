import pandas as pd
import keras
import matplotlib as plot
from keras import layers, callbacks

concrete_filepath = './Datasets/concrete.csv'

concrete_filedata = pd.read_csv(concrete_filepath)

concrete_filedata.fillna(0)

df_train = concrete_filedata.sample(frac=0.7, random_state=0)
df_valid = concrete_filedata.drop(df_train.index)

# Scale |0,1|
# Mapping the values and ensuring the don't exceed the range of 0-1
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

train_x = df_train.drop('CompressiveStrength', axis=1)
valid_x = df_valid.drop('CompressiveStrength', axis=1)
train_y = df_train['CompressiveStrength']
valid_y = df_valid['CompressiveStrength']

cols, rows = train_x.shape

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Input(shape=(rows,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.3), # Apply a 30% dropout to the next layer
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.3), # Apply a 30% dropout to the next layer
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.3), # Apply a 30% dropout to the next layer
    layers.BatchNormalization(),
    layers.Dense(1)
])

model.compile(
    keras.optimizers.Adam(),
    loss='mae'
)

history = model.fit(
    train_x, train_y,
    validation_data=(valid_x, valid_y),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print(f"minimum validation loss: {format(history_df['val_loss'].min())}")