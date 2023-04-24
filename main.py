import os
import torch
import torch.nn as nn
import json
from src.helpers import tokenize
from src.dataloader import get_dataloader
from src.custom_word2vec import CBOWModel, SkipGramModel
from src.trainer import Trainer
from src.metric_monitor import MetricMonitor

# Set path for dataset
TEXT_PATH = os.path.join("dataset", "text8.txt")

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Max vocabulary size
MAX_VOCAB_SIZE = 5000

# Number of epochs
EPOCHS = 5

# Model type
MODEL_TYPE = "cbow"  # or "skipgram"

# Embedding (vector) size
EMBEDDING_SIZE = 100

# Save path
SAVE_PATH = "results"


if __name__ == "__main__":
    # Read input text, tokenize and build vocabulary
    with open(TEXT_PATH, "r") as f:
        raw_txt = f.read()

    vocab, tokens = tokenize(inp=raw_txt, vocab_size=MAX_VOCAB_SIZE, default_token="<unk>")
    VOCAB_SIZE = min(MAX_VOCAB_SIZE, vocab.get_size())

    # Dataloaders
    train_dataloader = get_dataloader(
        tokens=tokens,
        model_type=MODEL_TYPE,
        loader_type="train",
        vocab=vocab
    )
    val_dataloader = get_dataloader(
        tokens=tokens,
        model_type=MODEL_TYPE,
        loader_type="val",
        vocab=vocab
    )

    if MODEL_TYPE == "cbow":
        model = CBOWModel(vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE)
    elif MODEL_TYPE == "skipgram":
        model = SkipGramModel(vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE)
    else:
        raise NotImplementedError

    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    metric_monitor = MetricMonitor(
        epochs=EPOCHS
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        metric_monitor=metric_monitor,
        epochs=EPOCHS
    )

    trainer.train()

    # Save vocabulary
    vocab_path = os.path.join(SAVE_PATH, MODEL_TYPE, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab.get_stoi(), f)

    # Save tokens
    tokens_path = os.path.join(SAVE_PATH, MODEL_TYPE, "tokens.txt")
    with open(tokens_path, "w") as f:
        f.write(" ".join(tokens))

    # Save metrics
    metrics_path = os.path.join(SAVE_PATH, MODEL_TYPE, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metric_monitor.metrics, f)

    # Save model
    model_path = os.path.join(SAVE_PATH, MODEL_TYPE, "model.pth")
    torch.save(model, model_path)

    # Save model's weights
    model_w_path = os.path.join(SAVE_PATH, MODEL_TYPE, "model_state.pth")
    torch.save(model.state_dict(), model_w_path)