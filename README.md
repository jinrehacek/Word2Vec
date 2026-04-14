# Word2Vec (with PyTorch)
## Overview
Word2Vec is a machine-learning model that creates latent space of vectors representing tokens (words) from given data corpus. This model was a breakthrough in ML and retains interesting properties as vectors represent syntactic and semantic context/relationships.

Papers: 
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546)

## Data
- Training corpus: Shakespeare *(complete works)* + Dostoevsky *(Crime & Punishment, Brother Karamazov, Idiot)* + WikiText-2 (~2M tokens)
- Inputted as raw text (~ 20 MB), lowercased & tokenized with regex
- words appearing < 5 times and  50 most frequent words removed
- The training data aren't ideal as it's a combination of multiple styles of texts. Factual text or news would provide more useful connections than literature. 
- Using bigger dataset would require rewriting such that data is streamed, not processed at once

## Model
- Skip-gram with negative sampling
- Two embedding matrices (center and context) - words have different meanings when when they are only context to other words
- Embedded vectors have 100 dimensions
- Negative samples drawn from unigram distribution^(3/4) - *as per source paper*
- Loss: log-sigmoid objective from the paper

## Training
| Hyperparameter   | Value |
|------------------|-------|
| Embedding dim    |   100 |
| Window size      | 5     |
| Negative samples | 10    |
| Learning rate    | 1.0 (SGD) |
| Batch size       | 4096  |
| Epochs           | 100   |


## Results

### Nearest Neighbors

| Query | #1 | #2 | #3 |
|---|---|---|---|
| king | henry | richard | edward |
| queen | exeter | warwick | elizabeth |
| death | birth | wars | sun |
| daughter | heir | lady | isabella |
| wife | friend | dead | husband |
| money | kept | getting | tried |

### Analogies

| Analogy | Expected | Got |
|---|---|---|
| father - man + woman | mother | **mother** |
| brother - man + woman | sister | gentlemen  |
| men - man + woman | women | men |

Nearest neighbors capture similar topics well, for example "king" maps to English kings, "daughter" to female nobles and "death" to "birth." (I suppose present, but also chosen as examples, because of Shakespeare).
Analogies mostly fail: only 1 out of 3 gender analogies returns what was expected. 
The culprit is the size of training data as ~2M tokens across three (not very similar) sources is not enough for stable arithmetic

### Embedding Visualization
[PCA plot here]

<!-- TODO: rest of this and the PCA--> 
## What I'd Improve
- Larger single-domain corpus (WikiText-103)
- Subsampling instead of hard top-k removal
- Lower learning rate (0.025 per original paper)

## References
- Mikolov et al. 2013 — Distributed Representations of Words and Phrases
