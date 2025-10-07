# train.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
from config import *
from env import ContactRoutingEnv

# --- 1. Load & validate data ---
file_path = "../data/rl_calls_synthetic.csv"
df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
missing = [c for c in EXPECTED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=['first_utterance','best_queue','observed_aht_best'])
df['observed_aht_best'] = df['observed_aht_best'].astype(float)
df = df.drop(columns=['complexity_score'])

if (df['intent'].str.lower() == df['best_queue'].str.lower()).mean() >= 0.8:
    df = df.drop(columns=['intent'])
    cat_cols = ['channel','language','region']
else:
    cat_cols = ['channel','language','region','intent']

encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in cat_cols}
for col, le in encoders.items():
    df[col] = le.transform(df[col].astype(str))

num_cols = ['is_premium','prior_contacts_30d','sentiment']
df[num_cols] = df[num_cols].astype(float)

# --- 2. Generate embeddings ---
embedder = SentenceTransformer(MODEL_NAME)
embeddings = embedder.encode(df['first_utterance'].tolist(), show_progress_bar=True)
emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
df_emb = pd.DataFrame(embeddings, columns=emb_cols)
df = pd.concat([df.reset_index(drop=True), df_emb.reset_index(drop=True)], axis=1)

queues = sorted(df['best_queue'].unique())
queue2id = {q: i for i, q in enumerate(queues)}
df['queue_id'] = df['best_queue'].map(queue2id)

feature_cols = cat_cols + num_cols + emb_cols
X = df[feature_cols].astype(float).values
y = df['queue_id'].values
aht = df['observed_aht_best'].astype(float).values

# --- 3. Split ---
if RANDOM_SPLIT:
    X_train, X_test, y_train, y_test, aht_train, aht_test = train_test_split(X, y, aht, test_size=TEST_SIZE, random_state=42)
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp')
    split_point = int(len(df) * (1 - TEST_SIZE))
    train_df, test_df = df.iloc[:split_point], df.iloc[split_point:]
    X_train = train_df[feature_cols].values
    y_train = train_df['queue_id'].values
    aht_train = train_df['observed_aht_best'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['queue_id'].values
    aht_test = test_df['observed_aht_best'].values

# --- 4. Train & evaluate ---
env_train = ContactRoutingEnv(X_train, y_train, aht_train)
env_test  = ContactRoutingEnv(X_test,  y_test,  aht_test)

model = DQN("MlpPolicy", env_train, verbose=0, learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, train_freq=TRAIN_FREQ)
model.learn(total_timesteps=TOTAL_TIMESTEPS)

def evaluate(model, env, n_episodes=200):
    rewards, correct = [], 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if info["true_queue"] == info["chosen_queue"]:
            correct += 1
    return np.mean(rewards), correct / n_episodes

avg_reward, accuracy = evaluate(model, env_test)
print(f"Average Reward (test set): {avg_reward:.3f} | Routing Accuracy: {accuracy*100:.2f}%")