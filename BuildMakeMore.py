import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
words = open('names.txt', 'r').read().splitlines()
#print(words[:10])
#print(len(words))
#print(min(len(w) for w in words))
#print(max(len(w) for w in words))

# Empty list for bigrams
# b = {}
#
# for w in words:
#     chs = ['<S>'] + list(w) + ['<E>']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         bigram = (ch1, ch2)
#         b[bigram] = b.get(bigram, 0) + 1
#         #print(ch1, ch2)

#print(b)
#print(sorted(b.items(), key = lambda kv: -kv[1])) # Prints the tuples of keys and values
# key = lambda kv: kv[1] sorts by the count of the elements
# -kv prints most likely or tuples that occurred the most

# a = torch.zeros((3,5), dtype=torch.int32) # default float32
#print(a)
# a[1, 3] = 1
# a[0,0] = 5
# print(a)

# Creates an 27x27 array of zeros
N = torch.zeros((27,27), dtype=torch.int32)

# Construct char array
# set(''.join(words)) # sets dont allow duplicates
# print(set(''.join(words)))
# print(len(set(''.join(words))))
# print(sorted(list(set(''.join(words)))))

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # stoi (string to integer)
# print(stoi)
# stoi['<S>'] = 26 typical start <S> and end <E> token
# stoi['<E>'] = 27
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # just creates a reverse mapping integer, string
# print(itos)


for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
# print(N)

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range (27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
# plt.show()
# print(N[0, :])  # Gives 1 dimensional array of the first row

p = N[0].float()
p = p / p.sum()
# print(p)
# print(sum(p))

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# print(ix)
itos[ix]
# print(itos[ix])

# Determinalistic way of creating a Torch generator object
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
# print(p)

torch.multinomial(p, num_samples=20, replacement=True, generator=g)

# Prepare a matrix with the row of probabilities normalized to 1
# N + 1 is model smoothing which takes in account issues with zeros
P = (N+1).float()
P /= P.sum(1, keepdim=True)
# print(P[0].sum())

g = torch.Generator().manual_seed(2147483647)

for i in range(20):

    out = []
    ix = 0
    while True:

        p = P[ix]
        # "trained model" trained on just bigrams
        # p = N[ix].float()
        # p = p / p.sum()

        # "untrained" completely random model
        # p = torch.ones(27) / 27.0

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    # print(''.join(out))

# Goal: maximize likelihood of the data w.r.t model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)

log_likelihood = 0.0
n = 0

# for w in words:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n += 1
#         # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
#
# print(f'{log_likelihood=}')
# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll/n}')

# for w in words
# for w in ["joseph"]:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n += 1
#         print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
#
# print(f'{log_likelihood=}')
# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll / n}')

# Create the training set of bigrams (x,y)
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)
# print(xs, ys)

# Floats are feed into Neural Nets
# xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
# print(xenc)
# print(xenc.shape)
# plt.imshow(xenc)
# plt.show()
# print(xenc.dtype) # always check dtype

# W = torch.randn((27,27))
# print(W)
# logits = xenc @ W # predict log-counts
# @ is a matrix multiplier used in pytorch
# (5, 27) @ (27, 27) --> (5, 27)
# print((xenc @ W)[3,13]) # Dot product of 3rd input and 13 column of W matrix

# Softmax
# counts = logits.exp() # equivalent N
# probs = counts / counts.sum(1, keepdims=True) # probabilities for next character

# print(probs)
# print(probs.shape)
# print(probs[0].sum())
# print(probs[0])
# print(probs[0].shape)

# nlls = torch.zeros(5)
# for i in range(5):
#     # i-th bigram
#     x = xs[i].item() # input character index
#     y = ys[i].item() # label character index
#     print('----------')
#     print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x}, {y})')
#     print('input to the neural net:', x)
#     print('output probabilities from the neural net:', probs[i])
#     print('label (actual next character):', y)
#     p = probs[i, y]
#     print('probability assigned by the net to the correct character:', p.item())
#     logp = torch.log(p)
#     print('log likelihood:', logp.item())
#     nll = -logp
#     print('negative log likelihood:', nll.item())
#     nlls[i] = nll
#
# print('=========')
# print('average negative log likelihood, i.e. loss =', nlls.mean().item())


# ----- Optimization -----
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# Gradient Descent
for k in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()  # input to the network: one-hot encoding
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts, equivalent to N
    probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
    print(loss.item())

    # backward pass
    W.grad = None  # set to zero the gradient
    loss.backward()

    # update
    W.data += -50 * W.grad

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):

    out = []
    ix = 0
    while True:

        # ----------
        # BEFORE:
        # p = P[ix]
        # ----------
        # NOW:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True)  # probabilities for next character
        # ----------

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))