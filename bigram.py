import torch
import torch.nn as nn
from torch.nn import functional as F

#--hyperparameters--
batch_size = 64
block_size = 256
context_length = block_size
max_iters = 5000 
eval_interval = 500 
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 
n_embd = 384
num_heads = 6 
n_layer = 6
dropout = 0.2

#--read input file--
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f: 
    text = f.read()

#--encoder & decoder-- 
chars = sorted(set(list(text))) 
stoi =  {i : ch for ch, i in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(itos)
encode = lambda s: [stoi[char] for char in s]
decode = lambda s: ''.join([itos[ix] for ix in s])

#-- Train and eval split -- 
data = torch.tensor(encode(text), dtype=torch.long) 
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# data Loader
def get_batch(split): 
    data = {
        'train': train_data, 
        'val': val_data,
    }[split]
    ix = torch.randint(0, len(data) - context_length, (batch_size,)) 
    x = torch.stack([data[i: i+context_length] for i in ix])
    y = torch.stack([data[i+1: i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y 

# This is the evaluation mode. So you would like to tell pytorch to not compute gradients. 
@torch.no_grad()
def estimate_loss(): 
    out = {}
    # this is a good practice and it is not necessary in this case. This will set the model to evaluation mode. 
    # Some layers like batch norm or dropout behave differently in training and evaluation mode.
    model.eval()
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item() 
    model.train()
    return out

#Self attention 
class Head(nn.Module):
    def __init__(self, head_size): 
        super().__init__() 
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).bool())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        B , T , C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = k @ q.transpose(-2,-1) * (C ** -0.5)
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim= -1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# Multi head attention
class MultiHeadAttention(nn.Module): 
    def __init__(self, num_heads, head_size): 
        super().__init__() 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# feed forward 
class FeedForward(nn.Module): 
    def __init__(self, n_embd): 
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        self.out = self.net(x)
        return self.out

# Block 
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        # self.block = nn.Sequential(
        #     MultiHeadAttention(num_heads, n_embd//num_heads), 
        #     FeedForward(n_embd)
        # )
        # self.head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, n_embd//num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 


# simple language model 
class BigramLanguageModel(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range (n_layer)])
        self.ln_F = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target = None): 
        B, T = idx.shape 
        token_embd = self.token_embedding_table(idx) #(B,T,C)
        positional_embd = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = token_embd + positional_embd 
        x = self.blocks(x)
        x = self.ln_F(x)
        self.logits = self.lm_head(x)

        if target is None: 
            loss = None 
        else: 
            B, T, C = self.logits.shape
            self.logits = self.logits.view(B*T, C)
            target = target.view(-1)
            loss = F.cross_entropy(self.logits, target)
        return self.logits, loss
    
    def generate(self, idx, max_new_tokens = 100): 
        for _ in range(max_new_tokens): 
            idx_cond = idx[:, -block_size:]
            logits, loss  = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
    
model = BigramLanguageModel()
model = model.to(device)
print(sum(p.numel()) for p in model.parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


#--train loop--
for i in range(max_iters): 
    if i % eval_interval == 0: 
        losses = estimate_loss()
        print(f'step {i}: train loss {losses['train']:0.4f}, val loss {losses['val']:0.4f}')
    # sample data 
    xb, yb = get_batch('train')
    # forward pass
    logits, loss = model(xb, yb)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device) 
print(decode(model.generate(context, max_new_tokens = 100)[0].tolist()))