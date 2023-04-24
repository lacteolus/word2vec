import os
import torch
from collections import Counter
import torch.nn as nn
import numpy as np
from src.helpers import tokenize
from src.dataloader import get_dataloader
from src.custom_word2vec import CBOWModel, SkipGramModel
from src.trainer import Trainer
from src.metric_monitor import MetricMonitor

TEXT_PATH = os.path.join("dataset", "text8.txt")
# TEXT_PATH = os.path.join("dataset", "alice30.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1
MAX_VOCAB_SIZE = 5000


if __name__ == "__main__":
    with open(TEXT_PATH, "r") as f:
        raw_txt = f.read()

    vocab, tokens = tokenize(inp=raw_txt, vocab_size=MAX_VOCAB_SIZE, default_token="<unk>")
    vocab_size = vocab.get_size()

    print(f"Vocab size {vocab_size}")

    # train_dataloader = get_dataloader(
    #     tokens=tokens,
    #     model_type="cbow",
    #     loader_type="train",
    #     vocab=vocab
    # )
    # val_dataloader = get_dataloader(
    #     tokens=tokens,
    #     model_type="cbow",
    #     loader_type="val",
    #     vocab=vocab
    # )
    #
    # model = CBOWModel(vocab_size)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # metric_monitor = MetricMonitor(
    #     epochs=EPOCHS
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     device=DEVICE,
    #     metric_monitor=metric_monitor,
    #     epochs=EPOCHS
    # )
    #
    # trainer.train()

    # train_loss = []
    # val_loss = []
    #
    # for epoch in range(1, EPOCHS + 1):
    #     print(f"Epoch {epoch}/{EPOCHS}")
    #
    #     model.train()
    #     epoch_train_loss = []
    #     epoch_val_loss = []
    #
    #     # Train model
    #     for i, data in enumerate(train_dataloader):
    #         print(data)
    #         inputs = data[0].to(DEVICE)
    #         labels = data[1].to(DEVICE)
    #
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_train_loss.append(loss.item())
    #
    #         # Evaluate model
    #         model.eval()
    #         with torch.no_grad():
    #             for _, batch_data in enumerate(val_dataloader):
    #                 inputs = batch_data[0].to(DEVICE)
    #                 labels = batch_data[1].to(DEVICE)
    #
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)
    #
    #                 epoch_val_loss.append(loss.item())
    #
    #     train_loss.append(np.mean(epoch_train_loss))
    #     val_loss.append(np.mean(epoch_val_loss))
    #
    #     print(
    #         f"Train Loss={train_loss[-1]:.5f}, \n"
    #         f"Val Loss={val_loss[-1]:.5f}"
    #     )

