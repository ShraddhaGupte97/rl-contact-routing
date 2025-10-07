# config.py
MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_COLS = [
    'call_id','timestamp','channel','language','region','is_premium',
    'prior_contacts_30d','sentiment','intent','best_queue',
    'first_utterance','complexity_score','observed_aht_best'
]
RANDOM_SPLIT = True  # False → time-based split
TEST_SIZE = 0.2

# DQN hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TRAIN_FREQ = 4
TOTAL_TIMESTEPS = 2000