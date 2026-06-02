import torch
from torch.nn.functional import normalize
from main import SkipGram, VECTOR_LENGHT, VOCAB_SIZE, Word2Id, Id2Word


device = (
    torch.accelerator.current_accelerator().type  # pyright: ignore
    if torch.accelerator.is_available()
    else "cpu"
)


model = SkipGram(VOCAB_SIZE, VECTOR_LENGHT)
model.load_state_dict(
    torch.load("./model5.ptch", weights_only=True, map_location="cpu")
)

model.eval()


weights_1 = model.emb1.weight.data.cpu()
weights_2 = model.emb2.weight.data.cpu()

combined_embeddings = weights_1 + weights_2
results = normalize(combined_embeddings, p=2, dim=1)


def get_similar(to: str, n: int = 1):
    target_vector = results[Word2Id[to]]

    dotprodcuts = torch.matmul(target_vector, results.T)
    sim_scores, sim_vectors = torch.topk(dotprodcuts, k=n + 1)

    print(f"\n{n} most simil. words to {to}")
    for i in range(1, n + 1):
        print(Id2Word[sim_vectors[i].item()], sim_scores[i].item())


def AminusBplusC(a: str, b: str, c: str):
    va = results[Word2Id[a]]
    vb = results[Word2Id[b]]
    vc = results[Word2Id[c]]

    vd = (va - vb) + vc
    dotprodcuts = torch.matmul(vd, results.T)
    sim_scores, sim_vectors = torch.topk(dotprodcuts, k=3)
    print(f"\n{a}-{b}+{c} is: ")
    for i in range(3):
        print(Id2Word[sim_vectors[i].item()], sim_scores[i].item())


get_similar("king", n=3)
get_similar("queen", n=3)
get_similar("murder", n=3)
get_similar("crime", n=3)
get_similar("royalty", n=3)
get_similar("angel", n=3)

AminusBplusC("king", "man", "woman")
AminusBplusC("father", "man", "woman")
AminusBplusC("brother", "man", "woman")
AminusBplusC("uncle", "man", "woman")
AminusBplusC("boy", "man", "woman")
AminusBplusC("husband", "man", "woman")
# AminusBplusC("punishment", "crime", "sin")
# AminusBplusC("bad", "good", "love")
# AminusBplusC("death", "life", "joy")
# AminusBplusC("hate", "enemy", "friend")
# AminusBplusC("master", "rich", "poor")
# AminusBplusC("blood", "sword", "poison")
# AminusBplusC("said", "say", "think")
# AminusBplusC("men", "man", "woman")
# AminusBplusC("kings", "king", "queen")
# AminusBplusC("crown", "head", "chair")

get_similar("king", n=3)
get_similar("sword", n=3)
get_similar("death", n=3)
get_similar("god", n=3)
get_similar("money", n=3)
get_similar("horse", n=3)
get_similar("soldier", n=3)
get_similar("daughter", n=3)
get_similar("wife", n=3)
get_similar("husband", n=3)
