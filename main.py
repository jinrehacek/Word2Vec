# import random
# from math import sqrt
import re
from pathlib import Path

import torch
from torch import Tensor, nn, tensor
from torch.nn.functional import logsigmoid


# ---------------- CONSTANTS -------------------
data_path = Path("data/final-data.txt")
CONTEXT = 5  # how many words around of given word we are creating pairs
EPOCHS = 100  # num of training epochs
LEARNING_RATE = 1
VECTOR_LENGHT = 100  # dimensions of final vectors we get in the embedding
K_NEGATIVE_SAMPLES = 10  # how many negative samples we use in negative sampling
BATCH_SIZE = 4096
MIN_COUNT = 5  # how many times i have to see the word so i care
DELETE_K_TOP_WORDS = 50
# TRESHOLD = 1e-5  for subsampling (currently unused)


# ------------ INPUT & INIT ---------------------------
device = (
    torch.accelerator.current_accelerator().type  # pyright: ignore
    if torch.accelerator.is_available()
    else "cpu"
)

with open(data_path.absolute(), "r") as file:
    data_raw0 = file.read()

# get rid of weird/book apostrophe
data_raw1 = data_raw0.lower().replace("’", "'")

tokens_raw: list[str] = re.findall(r"[a-zA-Z']+", data_raw1)

# --------------------------------------------------------------------------------------------
#                   COUNT WORDS & REMOVE TOO RARE AND TOO ABUNDANT ONES
# --------------------------------------------------------------------------------------------
word_counts: dict[str, int] = dict()
for token in tokens_raw:
    if token in word_counts:
        word_counts[token] += 1
    else:
        word_counts[token] = 1

# sort by their presence
counts_sorted: list[tuple[str, int]] = sorted(
    word_counts.items(), key=lambda x: x[1], reverse=True
)
unwanted_words: set[str] = set([word for word, _ in counts_sorted[:DELETE_K_TOP_WORDS]])

# clean up tokens
tokens: list[str] = [
    token
    for token in tokens_raw
    if word_counts[token] >= MIN_COUNT and token not in unwanted_words
]
N_TOKENS = len(tokens)

# NOTE: I did not manage to get it working well with subsampling.
# That may have been due to bad implementation or hyperparameters.

# ---- SUBSAMPLING ---------
# filtered_tokens = []
# for token in tokens:
#     freq = word_counts[token] / N_TOKENS
#     drop_prob = max(0.0, 1.0 - sqrt(TRESHOLD / freq))
#     if random.random() > drop_prob:
#         filtered_tokens.append(token)
# tokens = filtered_tokens
# N_TOKENS = len(tokens)


# --------------------------------------------------------------------------------------------
#         CREATE SET OF UNIQUE WORDS & LOOKUP TABLES & DISTRIBUTION & PAIRS TENSOR
# --------------------------------------------------------------------------------------------
Word2Id: dict[str, int] = dict()
Id2Word: dict[int, str] = dict()
index = 0

unique_words = set()
for token in tokens:
    if token not in unique_words:
        Word2Id[token] = index
        Id2Word[index] = token

        unique_words.add(token)
        index += 1

VOCAB_SIZE = len(unique_words)

word_frequency = torch.zeros(VOCAB_SIZE)
for token in tokens:
    word_frequency[Word2Id[token]] += 1

# --------- DISTRIBUTION ---------------
token_distribution: Tensor = word_frequency.pow(0.75)
# normalise
token_distribution: Tensor = token_distribution / token_distribution.sum()


# --------- CREATE PAIRS ---------------
id_pairs: list[tuple[int, int]] = []
for i in range(N_TOKENS):
    for j in range(i - CONTEXT, i + CONTEXT + 1):
        if j >= 0 and j < N_TOKENS and i != j:
            # (CENTER, CONTEXT)
            id_pairs.append((Word2Id[tokens[i]], Word2Id[tokens[j]]))

big_tensor = tensor(id_pairs, dtype=torch.long, device=device)


def loss_func(positive_dp: Tensor, negative_dp: Tensor):
    """
    Input: the dot products for
     - positive_dp = vector_center * vector_context (dot product); [BATCH_SIZE] scalars
     - negative_dp = vector_center * vector_context (dot product); [BATCH_SIZE, K_NEGATIVE_SAMPLES] of scalars
    """
    pos_sigm = logsigmoid(positive_dp)
    # sum over the k scalars in negative_dp (after the logsigmoid, as per paper)
    neg_sigm = logsigmoid(-negative_dp).sum(dim=1)

    batch_loss = pos_sigm + neg_sigm  # tensor [BATCH_SIZE] of scalars
    return -batch_loss.mean()


class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, vector_lenght):
        super().__init__()
        # 2 layers for assymentry
        self.emb1 = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=vector_lenght
        )
        self.emb2 = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=vector_lenght
        )

        # init on sensible (small) value, so it doesnt get crushed by sigmoid near extremes
        init_value = 1.0 / vector_lenght
        nn.init.uniform_(self.emb1.weight, -init_value, init_value)
        nn.init.uniform_(self.emb2.weight, -init_value, init_value)

    def forward(self, center_word: Tensor, context_word: Tensor):
        center: Tensor = self.emb1(center_word)
        context: Tensor = self.emb2(context_word)

        # if we are doing negative_pass -> context_word is [BATCH_SIZE, K_NEGATIVE_SAMPLES, VECTOR_LENGHT]
        if context.dim() == 3:
            # adds new dim as 1st, aka after original 0th dim
            center = center.unsqueeze(1)
            # center becomes [BATCH_SIZE, 1, VECTOR_LENGHT]

        dotproducts: Tensor = (context * center).sum(dim=-1)
        return dotproducts


MODEL = SkipGram(vocabulary_size=VOCAB_SIZE, vector_lenght=VECTOR_LENGHT).to(
    device=device
)
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)


def training_loop(
    data_tensor: Tensor,
    model: SkipGram,
    optimizer,
    loss_func,
):
    total_loss = 0
    index = 0

    # data_tensor is [NUM_OF_PAIRS, 2]
    N_PAIRS = len(data_tensor)
    permutation = torch.randperm(N_PAIRS, device=device)
    shuffled: Tensor = torch.index_select(data_tensor, 0, permutation)

    for start in range(0, N_PAIRS, BATCH_SIZE):
        # in case we are near end
        next_start = min(N_PAIRS, start + BATCH_SIZE)
        current_batch_size = next_start - start

        center_batch = shuffled[start:next_start, 0]
        context_batch = shuffled[start:next_start, 1]

        positive_dp = model(center_batch, context_batch)

        negative_words = (
            torch.multinomial(
                token_distribution,
                num_samples=current_batch_size * K_NEGATIVE_SAMPLES,
                replacement=True,
            )
            .view(current_batch_size, K_NEGATIVE_SAMPLES)
            .to(device)
        )  # => [BATCH_SIZE, K_NEGATIVE_SAMPLES]

        negative_dp = model(center_batch, negative_words)
        loss = loss_func(positive_dp, negative_dp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # ------PRINTING---------
        index += 1
        print_every = 1000
        if index % print_every == 0:
            print(f"loss: {loss.item():.3f} at index: {index // print_every} ")

    return total_loss, index


if __name__ == "__main__":
    MODEL.train()
    for i in range(EPOCHS):
        print(f"-------------EPOCH {i + 1}/{EPOCHS}--------------")
        total_loss, index = training_loop(
            data_tensor=big_tensor,
            model=MODEL,
            optimizer=OPTIMIZER,
            loss_func=loss_func,
        )
        print(f"Average loss: {total_loss / index:.3f} ")
