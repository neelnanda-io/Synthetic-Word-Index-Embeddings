# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

# %%
SEED = 42
torch.set_grad_enabled(True)
# %%
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
common_words = open("common_words.txt", "r").read().split("\n")
print(common_words[:10])
# %%
num_tokens = [len(model.to_tokens(" "+word, prepend_bos=False).squeeze(0)) for word in common_words]
print(list(zip(num_tokens, common_words))[:10])
# %%
word_df = pd.DataFrame({"word": common_words, "num_tokens": num_tokens})
word_df = word_df.query('num_tokens < 4')
word_df.value_counts("num_tokens")
# %%
prefix = "The United States Declaration of Independence received its first formal public reading, in Philadelphia.\nWhen"
PREFIX_LENGTH = len(model.to_tokens(prefix, prepend_bos=True).squeeze(0))
NUM_WORDS = 7
MAX_WORD_LENGTH = 3
# %%
train_filter = np.random.rand(len(word_df)) < 0.8
train_word_df = word_df.iloc[train_filter]
test_word_df = word_df.iloc[~train_filter]
print(train_word_df.value_counts("num_tokens"))
print(test_word_df.value_counts("num_tokens"))
train_word_by_length_array = [np.array([" "+j for j in train_word_df[train_word_df.num_tokens==i].word.values]) for i in range(1, MAX_WORD_LENGTH+1)]
test_word_by_length_array = [np.array([" "+j for j in test_word_df[test_word_df.num_tokens==i].word.values]) for i in range(1, MAX_WORD_LENGTH+1)]
# %%


def gen_batch(batch_size, word_by_length_array):
    word_lengths = torch.randint(1, MAX_WORD_LENGTH+1, (batch_size, NUM_WORDS))
    words = []
    for i in range(batch_size):
        row = []
        for word_len in word_lengths[i].tolist():
            word = word_by_length_array[word_len-1][np.random.randint(len(word_by_length_array[word_len-1]))]
            row.append(word)
        words.append("".join(row))
    full_tokens = torch.ones((batch_size, PREFIX_LENGTH + MAX_WORD_LENGTH*NUM_WORDS), dtype=torch.int64)
    tokens = model.to_tokens([prefix+word for word in words], prepend_bos=True)
    full_tokens[:, :tokens.shape[-1]] = tokens
    
    first_token_indices = torch.concatenate([
        torch.zeros(batch_size, dtype=int)[:, None], word_lengths.cumsum(dim=-1)[..., :-1]
    ], dim=-1) + PREFIX_LENGTH
    
    last_token_indices = word_lengths.cumsum(dim=-1) - 1 + PREFIX_LENGTH
    return full_tokens, words, word_lengths, first_token_indices, last_token_indices
tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(10, train_word_by_length_array)
tokens, words, word_lengths, first_token_indices, last_token_indices
# %%
batch_size = 256
epochs = 1000

# first_token_probe = (torch.randn((n_layers//2, d_model, NUM_WORDS, )).cuda() / 3 / np.sqrt(d_model)).requires_grad_(True)
# first_token_opt = torch.optim.AdamW([first_token_probe], lr=1e-3, weight_decay=1e-2)
# last_token_probe = (torch.randn((n_layers//2, d_model, NUM_WORDS, )).cuda() / 3 / np.sqrt(d_model)).requires_grad_(True)
# last_token_opt = torch.optim.AdamW([last_token_probe], lr=1e-3, weight_decay=1e-2)

# def probe_loss(residuals, probe, token_indices):
#     probe_outs = einops.einsum(residuals[:, torch.arange(len(token_indices)).to(residuals.device)[:, None], token_indices, :], probe, "layer batch word d_model, layer d_model out_word -> layer batch word out_word")
#     probe_outs = probe_outs.log_softmax(dim=-1)
#     # return probe_outs
#     return probe_outs.diagonal(-1, -2).mean([-1, -2])

# tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(batch_size, train_word_by_length_array)
# with torch.no_grad():
#     _, cache = model.run_with_cache(tokens.cuda(), names_filter=lambda x: x.endswith("resid_post"))
#     residuals = cache.stack_activation("resid_post")[:n_layers//2]
# print(residuals.shape, tokens.shape, first_token_indices.shape, last_token_indices.shape)
# # %%
# first_token_losses_ewma = torch.tensor(1.3).cuda()
# last_token_losses_ewma = torch.tensor(1.3).cuda()
# ewma_beta = 0.95
# for i in tqdm.tqdm(range(epochs)):
#     tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(batch_size, train_word_by_length_array)
#     with torch.no_grad():
#         _, cache = model.run_with_cache(tokens.cuda(), names_filter=lambda x: x.endswith("resid_post"))
#     residuals = cache.stack_activation("resid_post")

#     first_token_opt.zero_grad()
#     first_token_losses = -probe_loss(residuals, first_token_probe, first_token_indices.cuda()).sum()
#     first_token_losses.backward()
#     first_token_opt.step()

#     last_token_opt.zero_grad()
#     last_token_losses = -probe_loss(residuals, last_token_probe, last_token_indices.cuda()).sum()
#     last_token_losses.backward()
#     last_token_opt.step()

#     first_token_losses_ewma = ewma_beta * first_token_losses_ewma + (1-ewma_beta) * first_token_losses.detach()
#     last_token_losses_ewma = ewma_beta * last_token_losses_ewma + (1-ewma_beta) * last_token_losses.detach()

#     if i%100 == 0:
#         print(first_token_losses_ewma/n_layers, last_token_losses_ewma/n_layers)
# %%
torch.set_grad_enabled(False)
epochs = 100
all_first_token_residuals = []
all_last_token_residuals = []
for i in tqdm.tqdm(range(epochs)):
    tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(batch_size, train_word_by_length_array)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.cuda(), names_filter=lambda x: x.endswith("resid_post"))
        residuals = cache.stack_activation("resid_post")
        first_token_residuals = residuals[:, torch.arange(len(first_token_indices)).to(residuals.device)[:, None], first_token_indices, :]
        last_token_residuals = residuals[:, torch.arange(len(last_token_indices)).to(residuals.device)[:, None], last_token_indices, :]
        # print(first_token_residuals.shape, last_token_residuals.shape)
        all_first_token_residuals.append(to_numpy(first_token_residuals))
        all_last_token_residuals.append(to_numpy(last_token_residuals))
all_first_token_residuals = np.concatenate(all_first_token_residuals, axis=1)
all_last_token_residuals = np.concatenate(all_last_token_residuals, axis=1)
print(all_first_token_residuals.shape)
print(all_last_token_residuals.shape)
# %%
layer = 3
resids = all_first_token_residuals[layer]
first_ave = resids[:5000, 0].mean(0)
second_ave = resids[:5000, 1].mean(0)
diff = first_ave - second_ave
px.box(to_numpy(resids[5000:] @ diff))

# %%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

LAYER = 3
y = np.array([j for i in range(len(all_first_token_residuals[0])) for j in range(NUM_WORDS)])
X = all_last_token_residuals[LAYER, :].reshape(-1, d_model)

# Assuming X is your input data and y are your labels
# X = np.random.rand(batch_size, 768)  # replace with your actual data
# y = np.random.randint(0, 3, batch_size)  # replace with your actual labels

# Split the dataset into train and test sets
indices = to_numpy(torch.randperm(len(X))[:10000])
X_train, X_test, y_train, y_test = train_test_split(X[indices], y[indices], test_size=0.1, random_state=42)

# Create a Logistic Regression model
lr_model = LogisticRegression(multi_class='ovr', solver='saga', random_state=42, max_iter=100, C=1.0)

# Fit the model on the training data
lr_model.fit(X_train, y_train)

# Predict the labels of the test set

# Print a classification report
y_pred = lr_model.predict(X_train)
print(classification_report(y_train, y_pred))
y_pred = lr_model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
test_batches = 10
LAYER = 3
last_token_predictions_list = []
last_token_abs_indices_list = []
with torch.no_grad():
    for i in tqdm.tqdm(range(test_batches)):
        tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(batch_size, test_word_by_length_array)
        _, cache = model.run_with_cache(tokens.cuda(), names_filter=lambda x: x.endswith("resid_post"))
        residuals = cache.stack_activation("resid_post")
        first_token_residuals = residuals[:, torch.arange(len(first_token_indices)).to(residuals.device)[:, None], first_token_indices, :]
        last_token_residuals = residuals[:, torch.arange(len(last_token_indices)).to(residuals.device)[:, None], last_token_indices, :]
        last_token_resids = to_numpy(einops.rearrange(last_token_residuals[LAYER], "batch word d_model -> (batch word) d_model"))
        last_token_predictions_list.append(lr_model.predict(last_token_resids))
        last_token_abs_indices_list.append(to_numpy(last_token_indices.flatten()))
    # print(first_token_residuals.shape, last_token_residuals.shape)
    # all_first_token_residuals.append(to_numpy(first_token_residuals))
    # all_last_token_residuals.append(to_numpy(last_token_residuals))
# %%
last_token_abs_indices = np.concatenate(last_token_abs_indices_list)
last_token_predictions = np.concatenate(last_token_predictions_list)

df = pd.DataFrame({
    "index": [i for _ in range(batch_size * test_batches) for i in range(NUM_WORDS)],
    "abs_pos": last_token_abs_indices,
    "pred": last_token_predictions,
})

# %%

px.histogram(df, x="abs_pos", color="pred", facet_row="index", barnorm="fraction").show()
# %%
