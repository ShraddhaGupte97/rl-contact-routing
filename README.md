# 🎯 RL Agent-Assist for Contact Routing

This project prototypes an offline **RL-based agent-assist system** that learns to route customer contacts to the best service queue — reducing transfers and improving handling time.  
The model uses customer transcript, context metadata, and a **Deep Q-Network (DQN)** agent to simulate routing optimization.

---

## ✨ Features

- **Reinforcement Learning (DQN)** agent built with `stable-baselines3`
- **Sentence embeddings** using `SentenceTransformers (MiniLM)`
- **Custom Gymnasium environment** (`ContactRoutingEnv`)
- **End-to-end pipeline:** data → embeddings → RL training → evaluation  

---

## 📁 Dataset

This project uses a **synthetic customer support dataset** and has been included here **for educational and portfolio purposes only**.  

Key columns include:

```
| Column              | Description                                                         |
|---------------------|---------------------------------------------------------------------|
| `first_utterance`   | Customer’s opening statement or problem description                 |
| `best_queue`        | True optimal routing queue (target action)                          |
| `sentiment`         | Sentiment score or label derived from utterance                     |
| `region`            | Customer’s geographic region                                        |
| `language`          | Language of interaction                                             |
| `channel`           | Contact medium (chat, email, call, etc.)                            |
| `observed_aht_best` | Actual handling time for the optimal route (used in reward shaping) |
| `timestamp`         | Synthetic interaction time for potential temporal splits            |
| `is_premium`        | Whether the customer is a premium or priority user                  |
| `prior_contacts_30d`| Number of customer contacts in the last 30 days                     |
```

---

## ⚙️ Pipeline Overview

### How It Works

The RL pipeline is designed for **offline learning**, where the agent learns optimal routing policies from historical data without live deployment.

1. **Data Load & Preprocessing** - Encodes categorical features (channel, language, region, etc.)
. Normalizes sentiment & contact metrics
2. **Text Embeddings** - Uses all-MiniLM-L6-v2 from SentenceTransformers to convert first utterances into dense vectors
3. **Gym Environment (ContactRoutingEnv)** - Each call = one episode. Reward = +1 − normalized AHT for correct queue, −1 − AHT otherwise
4️. **DQN Agent Training** - Learns to select best queue (action) for given state (contact features)
5. **Evaluation** - Reports average reward and routing accuracy on unseen test data

--- 

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/ShraddhaGupte97/rl-contact-routing.git
cd rl-agent-assist-routing
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Project
```bash
cd modular_version
python train_dqn.py
```

### 5. Example Output
```bash
Training DQN agent...
Average Reward (test set): 0.164 | Routing Accuracy: 93.00%
```
---

## 📁 Project Structure

rl-agent-assist-routing/
│
├── data/
│ └── rl_calls_synthetic.csv # synthetic dataset
│
├── modular_version/ # clean modular breakdown
│ ├── config.py
│ ├── env.py
│ └── train_dqn.py
│
├── requirements.txt # dependencies
└── README.md

---

##🧭 Next Steps

- Transition from Offline → Online RL - Extend the current offline-trained DQN to an online or hybrid RL setup where the agent continues learning from live customer interactions under safety constraints.

- Multi-Step Routing Decisions

- Reward Shaping & SLA-Aware Optimization - Tune reward functions to balance multiple KPIs such as transfer rate, handling time, and SLA adherence.

- Incorporate RLHF - Introduce human-in-the-loop feedback signals — e.g., QA evaluations, customer satisfaction scores, or agent annotations — to refine the policy beyond reward-based optimization, making the routing model more aligned with real user and business objectives.

---

## 👩‍💻 Author

**Shraddha Gupte**
*Data Scientist | Machine Learning | NLP | LLM | Product Strategy*
🔗 [LinkedIn](https://www.linkedin.com/in/shraddha-gupte/) | 🌐 [GitHub](https://github.com/shraddhagupte)
