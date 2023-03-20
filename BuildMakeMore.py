import torch
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

# Prepare a matrix with the row of probabilites normalized to 1
P = N.float()
P = P / P.sum(1, keepdim=True)

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
    print(''.join(out))