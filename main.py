from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Optional  # put this at top with other imports
from dddqn_model import DuelingDQN, PrioritizedReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi.middleware.cors import CORSMiddleware


from dddqn_model import DuelingDQN, PrioritizedReplayBuffer



# Global variables
uploaded_df = None
processed_df = None
X_train = X_test = y_train = y_test = None
scaler = None
feature_columns = None

# DDDQN globals
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dddqn_model = None
target_model = None
replay_buffer = None
dddqn_optimizer = None
gamma = 0.99
loss_fn = nn.MSELoss()





app = FastAPI(
    title="Smart City Cybersecurity API",
    description="Backend for detecting Safe / Threat traffic",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# just to test server
@app.get("/")
def root():
    return {"message": "Smart City Cybersecurity API is running 🚀"}

# ✅ upload CSV dataset
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    global uploaded_df

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()
    s = content.decode("utf-8")
    data = StringIO(s)
    df = pd.read_csv(data)

    uploaded_df = df   # <-- store dataset

    return {
        "message": "Dataset uploaded and stored successfully 🚀",
        "rows": df.shape[0],
        "columns": list(df.columns)
    }
@app.get("/preview")
def preview():
    global uploaded_df

    if uploaded_df is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet")

    return uploaded_df.head(5).to_dict()

    
@app.get("/preprocess")
def preprocess():
    global uploaded_df, processed_df

    if uploaded_df is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet")

    df = uploaded_df.copy()

    # 1) Drop rows with any missing values (simple for now)
    before_rows = df.shape[0]
    df = df.dropna()
    after_dropna_rows = df.shape[0]

    # 2) Select only numeric columns for IQR
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) == 0:
        raise HTTPException(status_code=400, detail="No numeric columns found for IQR")

    # 3) IQR outlier removal
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    # keep rows that are NOT outliers in any numeric column
    mask = ~(
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
        (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    df_clean = df[mask]
    after_iqr_rows = df_clean.shape[0]

    # store processed df
    processed_df = df_clean

    return {
        "message": "Preprocessing done ✅",
        "rows_before": int(before_rows),
        "rows_after_dropna": int(after_dropna_rows),
        "rows_after_iqr": int(after_iqr_rows),
        "numeric_columns_used": numeric_cols[:20]  # show first 20 numeric cols
    }

@app.get("/prepare-data")
def prepare_data():
    global processed_df, X_train, X_test, y_train, y_test, scaler, feature_columns

    if processed_df is None:
        raise HTTPException(status_code=404, detail="Please run /preprocess first")

    df = processed_df.copy()

    possible_labels = ["label", "attack", "target", "class", "malicious"]
    label_col = None
    for col in df.columns:
        if col.lower() in possible_labels:
            label_col = col
            break

    if label_col is None:
        raise HTTPException(
            status_code=400,
            detail="Could not find label column. Please make sure your dataset has a column named one of: label, attack, target, class, malicious"
        )

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    if X.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No numeric feature columns left after selection")

    # ✅ store feature column names
    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return {
        "message": "Data prepared for model training ✅",
        "total_rows": int(df.shape[0]),
        "feature_count": int(X.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "label_column": label_col
    }

@app.get("/train-model")
def train_model():
    global X_train, X_test, y_train, y_test, model, model_accuracy

    if X_train is None or y_train is None:
        raise HTTPException(status_code=404, detail="Please run /prepare-data first")

    # create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_accuracy = acc

    return {
        "message": "Model trained successfully ✅",
        "test_accuracy": float(acc)
    }



@app.get("/predict-one")
def predict_one(index: int = 0):
    global X_test, y_test, model

    if model is None:
        raise HTTPException(status_code=404, detail="Please train the model first")

    if X_test is None or y_test is None:
        raise HTTPException(status_code=404, detail="Please run /prepare-data first")

    if index < 0 or index >= X_test.shape[0]:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {X_test.shape[0]-1}")

    x = X_test[index].reshape(1, -1)
    true_label = y_test.iloc[index]
    pred_label = model.predict(x)[0]

    return {
        "message": "Prediction done ✅",
        "index": index,
        "true_label": int(true_label),
        "predicted_label": int(pred_label),
        "meaning": "0 = Safe / Normal, 1 = Threat / Attack (you can adjust to your dataset)"
    }


@app.post("/predict-dddqn-file")
async def predict_dddqn_file(file: UploadFile = File(...)):
    global dddqn_model, scaler, feature_columns

    if dddqn_model is None:
        raise HTTPException(status_code=400, detail="Please run /train-dddqn first")

    if scaler is None or feature_columns is None:
        raise HTTPException(status_code=400, detail="Please run /prepare-data first")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    # read new CSV
    content = await file.read()
    s = content.decode("utf-8")
    data = StringIO(s)
    new_df = pd.read_csv(data)

    # drop label column if present
    for col in ["label", "attack", "target", "class", "malicious"]:
        if col in new_df.columns:
            new_df = new_df.drop(columns=[col])
            break

    # keep only training features
    missing_cols = [c for c in feature_columns if c not in new_df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns in input file: {missing_cols}"
        )

    X_new = new_df[feature_columns].fillna(0)
    X_new_scaled = scaler.transform(X_new)

    X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

    dddqn_model.eval()
    with torch.no_grad():
        q_values = dddqn_model(X_tensor)
        preds = torch.argmax(q_values, dim=1).cpu().numpy()

    new_df_preview = new_df.copy()
    new_df_preview["prediction"] = preds

    safe_count = int((preds == 0).sum())
    threat_count = int((preds == 1).sum())

    return {
        "message": "DDDQN file prediction completed ✅",
        "total_rows": int(len(preds)),
        "safe_count_label_0": safe_count,
        "threat_count_label_1": threat_count,
        "preview_first_5": new_df_preview.head(5).to_dict(),
        "meaning": "0 = Safe / Normal, 1 = Threat / Attack (adjust meaning if needed)"
    }


@app.get("/init-dddqn")
def init_ddqn():
    global feature_columns, dddqn_model, target_model, replay_buffer, dddqn_optimizer

    if feature_columns is None:
        raise HTTPException(
            status_code=400,
            detail="feature_columns is None. Please run /prepare-data first"
        )

    input_dim = len(feature_columns)
    output_dim = 2  # 0 = Safe, 1 = Threat

    # Create models
    dddqn_model = DuelingDQN(input_dim, output_dim).to(device)
    target_model = DuelingDQN(input_dim, output_dim).to(device)
    target_model.load_state_dict(dddqn_model.state_dict())
    target_model.eval()

    # Create replay buffer AND optimizer
    replay_buffer = PrioritizedReplayBuffer(capacity=20000)
    dddqn_optimizer = optim.Adam(dddqn_model.parameters(), lr=0.001)

    return {
        "message": "DDDQN initialized successfully 🧠🔥",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "optimizer_exists": dddqn_optimizer is not None
    }




@app.get("/train-dddqn")
def train_dddqn(epochs: int = 3, batch_size: int = 64):
    global X_train, y_train, X_test, y_test
    global dddqn_model, target_model, replay_buffer, dddqn_optimizer, gamma

    # 🔍 Debug: check what is missing
    debug = {
        "X_train_is_none": X_train is None,
        "y_train_is_none": y_train is None,
        "dddqn_model_is_none": dddqn_model is None,
        "target_model_is_none": target_model is None,
        "replay_buffer_is_none": replay_buffer is None,
        "dddqn_optimizer_is_none": dddqn_optimizer is None,
    }
    if any(debug.values()):
        return {
            "message": "Some required objects are None. Check debug info.",
            "debug": debug
        }

    # 1) Fill replay buffer
    X_np = np.array(X_train)
    y_np = np.array(y_train)

    replay_buffer.memory.clear()
    for i in range(len(X_np)):
        state = X_np[i]
        label = int(y_np[i])  # expected 0/1

        for action in [0, 1]:
            reward = 1.0 if action == label else -1.0
            next_state = state
            done = 1.0
            replay_buffer.push((state, action, reward, next_state, done))

    dddqn_model.train()
    total_steps = 0

    for epoch in range(epochs):
        if len(replay_buffer) < batch_size:
            break

        num_batches = min(200, len(replay_buffer) // batch_size)

        for _ in range(num_batches):
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            q_values = dddqn_model(states)
            q_values = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                next_q_online = dddqn_model(next_states)
                next_actions = torch.argmax(next_q_online, dim=1)
                next_q_target = target_model(next_states)
                next_q = next_q_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
                target_q = rewards + gamma * (1.0 - dones) * next_q

            loss = loss_fn(q_values, target_q)

            dddqn_optimizer.zero_grad()
            loss.backward()
            dddqn_optimizer.step()

            total_steps += 1
            if total_steps % 50 == 0:
                target_model.load_state_dict(dddqn_model.state_dict())

    # Evaluate as classifier
    if X_test is not None and y_test is not None:
        dddqn_model.eval()
        with torch.no_grad():
            X_test_np = np.array(X_test)
            X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
            q_test = dddqn_model(X_test_tensor)
            preds = torch.argmax(q_test, dim=1).cpu().numpy()
        y_true = np.array(y_test)
        acc = accuracy_score(y_true, preds)
    else:
        acc = None

    return {
        "message": "DDDQN training completed ✅",
        "epochs": epochs,
        "batch_size": batch_size,
        "replay_buffer_size": len(replay_buffer),
        "test_accuracy_approx": float(acc) if acc is not None else None
    }



@app.get("/predict-dddqn-one")
def predict_dddqn_one(index: int = 0):
    global X_test, y_test, dddqn_model

    if dddqn_model is None:
        raise HTTPException(status_code=400, detail="Please run /train-dddqn first")

    if X_test is None or y_test is None:
        raise HTTPException(status_code=400, detail="Please run /prepare-data first")

    if index < 0 or index >= X_test.shape[0]:
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {X_test.shape[0]-1}")

    # take one sample from test set
    x = np.array(X_test[index]).reshape(1, -1)
    true_label = int(y_test.iloc[index])

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    dddqn_model.eval()
    with torch.no_grad():
        q_values = dddqn_model(x_tensor)
        pred_label = int(torch.argmax(q_values, dim=1).cpu().item())

    return {
        "message": "DDDQN prediction done ✅",
        "index": index,
        "true_label": true_label,
        "predicted_label": pred_label,
        "meaning": "0 = Safe / Normal, 1 = Threat / Attack (adjust based on your dataset)"
    }
