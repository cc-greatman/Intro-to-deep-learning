import pandas as pd
import keras
from keras import layers, callbacks
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

hotel_filepath = './Datasets/hotel.csv'

hotel_filedata = pd.read_csv(hotel_filepath)

x = hotel_filedata.copy()
y = x.pop('is_canceled')

x['arrival_date_month'] = x['arrival_date_month'].map({
    'January':1, 'February':2, 'March':3, 'April':4,
    'May':5, 'June':6, 'July':7, 'August':8, 'September':9,
    'October':10, 'November':11, 'December':12
})

features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr"
]

features_cat = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type",
    "customer_type"
]

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"),
    StandardScaler()
)

transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat)
)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, stratify=y, train_size=0.75)

train_x = preprocessor.fit_transform(train_x)
valid_x = preprocessor.transform(valid_x)

rows = train_x.shape[1]

model = keras.Sequential([
    layers.Input(shape=(rows,)),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
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
    epochs=200,
    batch_size=512,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss', 'val_loss']].plot(title="Cross-Entropy")
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

print(f"Best Validation Loss: {history_df['val_loss'].min():.4f} +\
      \nBest Accuracy loss: {history_df['val_binary_accuracy'].max():.4f}")