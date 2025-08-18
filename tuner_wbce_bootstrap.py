# Re-import necessary packages after state reset
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Concatenate, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, average_precision_score
from sklearn.utils import resample
from collections import Counter
from keras_tuner import Objective

print("WBCE Tuner - Curated")

# Create directory for logs
directory = "LSTM_tuning_logs_wbce_6/LSTM_hypo_6"
os.makedirs(directory, exist_ok=True)

# Define input dimensions
n_timesteps_meds = 30
n_features_meds = 33

n_timesteps_labs = 30
n_features_labs = 13

n_features_static = 65

n_timesteps_diet = 30
n_features_diet = 4 

n_timesteps_meal = 30
n_features_meal = 1

def weighted_binary_crossentropy(pos_weight, neg_weight):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = - (pos_weight * y_true * tf.math.log(y_pred) +
                  neg_weight * (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)
    return loss_fn
    
def binary_focal_loss(gamma=1.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), K.epsilon(), 1. - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)

        alpha_f = tf.cast(alpha, tf.float32)
        gamma_f = tf.cast(gamma, tf.float32)

        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), alpha_f, 1 - alpha_f)
        loss = -alpha_t * tf.pow(1. - pt, gamma_f) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return focal_loss

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score_keras(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

class F1ScoreCGPT(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score_cgpt', **kwargs):  # <-- FIX name here
        super(F1ScoreCGPT, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

batch_size = 4096 
print('batch size', batch_size)

# Define a generator to read training data in chunks
def train_generator(batch_size):
    meds_chunker = pd.read_csv("X_train_meds.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    labs_chunker = pd.read_csv("X_train_labs.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    diet_chunker = pd.read_csv("X_train_diet.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    meal_chunker = pd.read_csv("X_train_meal.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    static_chunker = pd.read_csv("X_train_static.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    output_chunker = pd.read_csv("y_train.csv.gz", chunksize=batch_size, header=None, compression='gzip')

    for i, (mchunk, lchunk, dchunk, mealchunk, schunk, ychunk) in enumerate(zip(meds_chunker, labs_chunker, diet_chunker, meal_chunker, static_chunker, output_chunker)):
        inputs = {
            "TIMESERIES_INPUT_meds": mchunk.values.reshape(-1, n_timesteps_meds, n_features_meds).astype(np.float32),
            "TIMESERIES_INPUT_labs": lchunk.values.reshape(-1, n_timesteps_labs, n_features_labs).astype(np.float32),
            "TIMESERIES_INPUT_diet": dchunk.values.reshape(-1, n_timesteps_diet, n_features_diet).astype(np.float32),
            "TIMESERIES_INPUT_meal": mealchunk.values.reshape(-1, n_timesteps_meal, n_features_meal).astype(np.float32),
            "STATIC_INPUT": schunk.values.astype(np.float32)
        }
        labels = ychunk.values.astype(np.float32).reshape(-1, 1)

        # NaN/Inf check
        if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in list(inputs.values()) + [labels]):
            print(f"[Train] Skipping batch {i} due to NaNs or Infs")
            continue

        yield inputs, labels

            
def val_generator(batch_size):
    meds_chunker = pd.read_csv("X_val_meds.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    labs_chunker = pd.read_csv("X_val_labs.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    diet_chunker = pd.read_csv("X_val_diet.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    meal_chunker = pd.read_csv("X_val_meal.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    static_chunker = pd.read_csv("X_val_static.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    output_chunker = pd.read_csv("y_val.csv.gz", chunksize=batch_size, header=None, compression='gzip')

    for i, (mchunk, lchunk, dchunk, mealchunk, schunk, ychunk) in enumerate(zip(meds_chunker, labs_chunker, diet_chunker, meal_chunker, static_chunker, output_chunker)):
        inputs = {
            "TIMESERIES_INPUT_meds": mchunk.values.reshape(-1, n_timesteps_meds, n_features_meds).astype(np.float32),
            "TIMESERIES_INPUT_labs": lchunk.values.reshape(-1, n_timesteps_labs, n_features_labs).astype(np.float32),
            "TIMESERIES_INPUT_diet": dchunk.values.reshape(-1, n_timesteps_diet, n_features_diet).astype(np.float32),
            "TIMESERIES_INPUT_meal": mealchunk.values.reshape(-1, n_timesteps_meal, n_features_meal).astype(np.float32),
            "STATIC_INPUT": schunk.values.astype(np.float32)
        }
        labels = ychunk.values.astype(np.float32).reshape(-1, 1)

        # NaN/Inf check
        if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in list(inputs.values()) + [labels]):
            print(f"[Val] Skipping batch {i} due to NaNs or Infs")
            continue

        yield inputs, labels


# Convert generator into TensorFlow Dataset
def dataset_from_generator(generator_func, batch_size):
    return tf.data.Dataset.from_generator(
        lambda: generator_func(batch_size),
        output_signature=(
            {
                "TIMESERIES_INPUT_meds": tf.TensorSpec(shape=(None, n_timesteps_meds, n_features_meds), dtype=tf.float32),
                "TIMESERIES_INPUT_labs": tf.TensorSpec(shape=(None, n_timesteps_labs, n_features_labs), dtype=tf.float32),
                "TIMESERIES_INPUT_diet": tf.TensorSpec(shape=(None, n_timesteps_diet, n_features_diet), dtype=tf.float32),
                "TIMESERIES_INPUT_meal": tf.TensorSpec(shape=(None, n_timesteps_meal, n_features_meal), dtype=tf.float32),
                "STATIC_INPUT": tf.TensorSpec(shape=(None, n_features_static), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        )
    )

num_train_samples = len(pd.read_csv("y_train.csv.gz", header=None, compression='gzip'))
num_val_samples = len(pd.read_csv("y_val.csv.gz", header=None, compression='gzip'))

print(f"Number of training samples: {num_train_samples}")
print(f"Number of validation samples: {num_val_samples}")

def build_model(hp):
    dense_units = hp.Int('dense_units', min_value=64, max_value=96, step=8)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=5e-4, sampling='LOG')
    num_layers = hp.Int('num_lstm_layers', 1, 2)
    for i in range(1, 4):
        if i == 1:
            hp.Int(f'lstm_units_layer_{i}', min_value=16, max_value=256, step=8)  
            hp.Float(f'dropout_rate_layer_{i}', min_value=0.5, max_value=0.9, step=0.1)  
        elif i == 2:
            hp.Int(f'lstm_units_layer_{i}', min_value=16, max_value=256, step=8)  
            hp.Float(f'dropout_rate_layer_{i}', min_value=0.5, max_value=0.9, step=0.1) 
        elif i == 3:
            hp.Int(f'lstm_units_layer_{i}', min_value=16, max_value=256, step=8)  
            hp.Float(f'dropout_rate_layer_{i}', min_value=0.5, max_value=0.9, step=0.1)  
	# Inputs
    recurrent_input_meds = tf.keras.Input(shape=(n_timesteps_meds, n_features_meds), name="TIMESERIES_INPUT_meds")
    recurrent_input_labs = tf.keras.Input(shape=(n_timesteps_labs, n_features_labs), name="TIMESERIES_INPUT_labs")
    recurrent_input_diet = tf.keras.Input(shape=(n_timesteps_diet, n_features_diet), name="TIMESERIES_INPUT_diet")
    recurrent_input_meal = tf.keras.Input(shape=(n_timesteps_meal, n_features_meal), name="TIMESERIES_INPUT_meal")
    static_input = tf.keras.Input(shape=(n_features_static,), name="STATIC_INPUT")

	# LSTM block
    def lstm_block(x):
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    hp.get(f'lstm_units_layer_{i+1}'), return_sequences=return_seq
                )
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(
                hp.get(f'dropout_rate_layer_{i+1}')
            )(x)
        return x

    # Process each timeseries input
    meds_out = lstm_block(recurrent_input_meds)
    labs_out = lstm_block(recurrent_input_labs)
    diet_out = lstm_block(recurrent_input_diet)
    meal_out = lstm_block(recurrent_input_meal)

    static_out = tf.keras.layers.Dense(dense_units, activation='relu')(static_input)
    static_out = tf.keras.layers.Dropout(0.1)(static_out)

    combined = tf.keras.layers.Concatenate()([meds_out, labs_out, diet_out, meal_out, static_out])
    dense_combined = tf.keras.layers.Dense(dense_units, activation='relu')(combined)
    dense_combined = tf.keras.layers.Dropout(0.1)(dense_combined)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_combined)

    model = tf.keras.Model(inputs=[
        recurrent_input_meds,
        recurrent_input_labs,
        recurrent_input_diet,
        recurrent_input_meal,
        static_input
    ], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=weighted_binary_crossentropy(pos_weight=pos_weight, neg_weight=neg_weight),
        metrics=[tf.keras.metrics.AUC(name='AUPR', curve='PR'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 F1ScoreCGPT(name='f1_score_cgpt')]
    )

    return model

y = pd.read_csv("y_train.csv.gz", header=None, compression='gzip').values.flatten()
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
print("Class Weights:", class_weight_dict)

# Assign weights
pos_weight = class_weight_dict[1]  # for y = 1
neg_weight = class_weight_dict[0]  # for y = 0

# Setup tuner
tuner = BayesianOptimization(
    build_model,
    objective=Objective("val_f1_score_cgpt", direction="max"),
    max_trials=10, #modify
    executions_per_trial=1,
    directory='LSTM_tuning_logs_wbce_6',
    project_name='LSTM_hypo_6' )

# Load datasets
train_dataset = dataset_from_generator(train_generator, batch_size).repeat().prefetch(tf.data.AUTOTUNE)
val_dataset = dataset_from_generator(val_generator, batch_size).repeat().prefetch(tf.data.AUTOTUNE)

steps_per_epoch = num_train_samples // batch_size
validation_steps = num_val_samples // batch_size

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Define early stopping callback
early_stop = EarlyStopping(
    monitor='val_f1_score_cgpt', 
    mode='max',  
    patience=3,
    restore_best_weights=True,
    verbose=1)

# Add to tuner.search
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=[early_stop],
    verbose=2)

# Get best model and parameters
best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best Hyperparameters:", best_hp.values)

steps_per_epoch = num_train_samples // batch_size
validation_steps = num_val_samples // batch_size

# Rebuild and retrain the best model on full train+val
best_model = build_model(best_hp)
# Define early stopping
early_stop = EarlyStopping(
    monitor='val_f1_score_cgpt',  # adjust to your metric
    mode='max',
    patience=3,
    restore_best_weights=True,
    verbose=1)

# Fit the model with early stopping
history = best_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
    callbacks=[early_stop])  

# Load test data and reshape
def test_generator(batch_size):
    meds_chunker = pd.read_csv("X_test_meds.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    labs_chunker = pd.read_csv("X_test_labs.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    diet_chunker = pd.read_csv("X_test_diet.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    meal_chunker = pd.read_csv("X_test_meal.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    static_chunker = pd.read_csv("X_test_static.csv.gz", chunksize=batch_size, header=None, compression='gzip')
    output_chunker = pd.read_csv("y_test.csv.gz", chunksize=batch_size, header=None, compression='gzip')

    for i, (mchunk, lchunk, dchunk, mealchunk, schunk, ychunk) in enumerate(zip(
        meds_chunker, labs_chunker, diet_chunker, meal_chunker, static_chunker, output_chunker)):

        inputs = {
            "TIMESERIES_INPUT_meds": mchunk.values.reshape(-1, n_timesteps_meds, n_features_meds).astype(np.float32),
            "TIMESERIES_INPUT_labs": lchunk.values.reshape(-1, n_timesteps_labs, n_features_labs).astype(np.float32),
            "TIMESERIES_INPUT_diet": dchunk.values.reshape(-1, n_timesteps_diet, n_features_diet).astype(np.float32),
            "TIMESERIES_INPUT_meal": mealchunk.values.reshape(-1, n_timesteps_meal, 1).astype(np.float32),
            "STATIC_INPUT": schunk.values.astype(np.float32)
        }
        labels = ychunk.values.astype(np.float32).reshape(-1, 1)

        if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in list(inputs.values()) + [labels]):
            print(f"[Test] Skipping batch {i} due to NaNs or Infs")
            continue

        yield inputs, labels
        
        
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator(batch_size),
    output_signature=(
        {
            "TIMESERIES_INPUT_meds": tf.TensorSpec(shape=(None, n_timesteps_meds, n_features_meds), dtype=tf.float32),
            "TIMESERIES_INPUT_labs": tf.TensorSpec(shape=(None, n_timesteps_labs, n_features_labs), dtype=tf.float32),
            "TIMESERIES_INPUT_diet": tf.TensorSpec(shape=(None, n_timesteps_diet, n_features_diet), dtype=tf.float32),
            "TIMESERIES_INPUT_meal": tf.TensorSpec(shape=(None, n_timesteps_diet, 1), dtype=tf.float32),
            "STATIC_INPUT": tf.TensorSpec(shape=(None, n_features_static), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE)

y_pred_probs = best_model.predict(test_dataset)

pd.DataFrame({"y_pred_prob": y_pred_probs.flatten()}).to_csv("wbce_y_pred_probs.csv", index=False)

# Define thresholds for evaluation
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

# Number of bootstrap samples
n_bootstrap = 100 #modify

# Store bootstrap results
bootstrap_results = {thresh: {"precision": [], "recall": [], "f1": [], "aupr": []} for thresh in thresholds}

y_test = np.loadtxt('y_test.csv.gz', delimiter=',')

bootstrap_size = int(0.5 * len(y_test))  # 50% of the test set

# Perform bootstrap sampling
for i in range(n_bootstrap):
    indices = np.random.choice(np.arange(len(y_test)), size=bootstrap_size, replace=True)
    y_true_sample = y_test[indices]
    y_pred_probs_sample = y_pred_probs[indices]

    for thresh in thresholds:
        y_pred_sample = (y_pred_probs_sample >= thresh).astype(int)
        precision = precision_score(y_true_sample, y_pred_sample, zero_division=0)
        recall = recall_score(y_true_sample, y_pred_sample, zero_division=0)
        f1 = f1_score(y_true_sample, y_pred_sample, zero_division=0)
		
        try:
            aupr = average_precision_score(y_true_sample, y_pred_probs_sample)
        except ValueError:
            aupr = np.nan  # Handle edge case if AUPR cannot be computed
            
        bootstrap_results[thresh]["precision"].append(precision)
        bootstrap_results[thresh]["recall"].append(recall)
        bootstrap_results[thresh]["f1"].append(f1)
        bootstrap_results[thresh]["aupr"].append(aupr)

# Compute 95% pivot confidence intervals
ci_summary = []

for thresh in thresholds:
    for metric in ["precision", "recall", "f1", "aupr"]:
        values = np.array(bootstrap_results[thresh][metric])
        lower = 2 * np.mean(values) - np.percentile(values, 97.5)
        upper = 2 * np.mean(values) - np.percentile(values, 2.5)
        ci_summary.append({
            "Threshold": thresh,
            "Metric": metric,
            "Mean": np.mean(values),
            "CI Lower": lower,
            "CI Upper": upper
        })

ci_df = pd.DataFrame(ci_summary)
print(ci_df)

records = []
for thresh, metrics in bootstrap_results.items():
    for p, r, f, a in zip(metrics["precision"], metrics["recall"], metrics["f1"], metrics["aupr"]):
        records.append({"threshold": thresh, "precision": p, "recall": r, "f1": f, "aupr": a})

bootstrap_df = pd.DataFrame(records)
bootstrap_df.to_csv("wbce_bootstrap_metrics.csv", index=False)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.savefig('wbce_train_val_loss.svg', bbox_inches='tight')