import re
from pathlib import Path

# TORCH imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn, tensor
from torch.nn.functional import logsigmoid

data_path = Path("data.txt")
CONTEXT = 5  # how many words around of given word we are creating pairs
EPOCHS = 100  # num of training epochs
LEARNING_RATE = 1e-2
VECTOR_LENGHT = 100  # dimensions of final vectors we get in the embedding
K_NEGATIVE_SAMPLES = 5  # how many negative samples we use in negative sampling
BATCH_SIZE = 4096

device = (
    torch.accelerator.current_accelerator().type  # pyright: ignore
    if torch.accelerator.is_available()
    else "cpu"
)

with open(data_path.absolute(), "r") as file:
    data_raw0 = file.read()

# get rid of weird apostrophe
data_raw1 = data_raw0.lower().replace("’", "'")

# tokenize
d_tokens = re.findall(r"[a-zA-Z']+", data_raw1)
N_TOKENS = len(d_tokens)

# --------- CREATE SET OF UNIQUE WORDS ---------------
unique_words = set()
for token in d_tokens:
    if token not in unique_words:
        unique_words.add(token)
VOCAB_SIZE = len(unique_words)


# --------- CREATE LOOKUP TABLES ---------------
Word2Id: dict[str, int] = dict()
Id2Word: dict[int, str] = dict()
index = 0
while len(unique_words) > 0:
    word = unique_words.pop()
    Word2Id[word] = index
    Id2Word[index] = word
    index += 1


# --------- CREATE PAIRS ---------------
id_pairs: list[tuple[int, int]] = []
for i in range(N_TOKENS):
    for j in range(i - CONTEXT, i + CONTEXT + 1):
        if j >= 0 and j < N_TOKENS and i != j:
            #                   CENTER;    CONTEXT
            id_pairs.append((Word2Id[d_tokens[i]], Word2Id[d_tokens[j]]))


# --------- CREATE DATALOADER ---------------
big_tensor = tensor(id_pairs, dtype=torch.long)
trainind_dataset = TensorDataset(big_tensor[:, 0], big_tensor[:, 1])
trainig_loader = DataLoader(
    dataset=trainind_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


def loss_func(positive_dp: Tensor, negative_dp: Tensor):
    """
    doesnt implement picking from uniform distributuins
    you need to suuply the vectors to be picked outsiede of it
    """
    # positive_dp is [1024]
    pos_sigm = logsigmoid(positive_dp)

    # negative_dp is [1024, 5]. We sum across the 5 negative samples (dim=1)
    # so we get a single negative penalty for each of the 1024 center words.
    neg_sigm = logsigmoid(-negative_dp).sum(dim=1)

    # Add them together. batch_loss is now an array of 1024 individual losses.
    batch_loss = pos_sigm + neg_sigm

    # Collapse the 1024 losses into a single average number (a scalar)
    return -batch_loss.mean()


class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, vector_lenght):
        super().__init__()
        self.emb1 = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=vector_lenght
        )
        self.emb2 = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=vector_lenght
        )

    def forward(self, center_word: Tensor, context_word: Tensor):
        center: Tensor = self.emb1(center_word)
        context: Tensor = self.emb2(context_word)

        if context.dim() == 3:
            center = center.unsqueeze(1)

        thing: Tensor = (context * center).sum(dim=-1)
        return thing


MODEL = SkipGram(vocabulary_size=VOCAB_SIZE, vector_lenght=VECTOR_LENGHT).to(
    device=device
)
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)


def training_loop(
    dataloader: DataLoader,
    model: SkipGram,
    optimizer,
    loss_func,
    k_negative_samples: int = 5,
):
    total_loss = 0
    index = 0
    for center_batch, context_bathc in dataloader:
        center_batch = center_batch.to(device)
        context_bathc = context_bathc.to(device)

        positive = model(center_batch, context_bathc)

        current_b_size = center_batch.size(0)

        # get k random IDs correspodning to some words as one vector
        neg_words = torch.randint(
            0, VOCAB_SIZE, (current_b_size, k_negative_samples), device=device
        )
        negative_dp = model(center_batch, neg_words)
        loss = loss_func(positive, negative_dp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        index += 1
        if index % 1000 == 0:
            print(f"loss: {loss.item():.4f} at index: {index // 1000} ")

    return total_loss


for i in range(EPOCHS):
    print(f"-------------EPOCH {i + 1}/{EPOCHS}--------------")
    total_loss = training_loop(
        dataloader=trainig_loader,
        model=MODEL,
        optimizer=OPTIMIZER,
        loss_func=loss_func,
        k_negative_samples=K_NEGATIVE_SAMPLES,
    )
    print(f"Average loss: {total_loss:.4f} ")
