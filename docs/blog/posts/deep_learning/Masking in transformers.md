---
title: Masking in transformers
date: 2024-06-13
draft: false
---
Masking is pretty fundamental to how transformers work. The main purpose of masking is to make sure the model doesn't "attend to" some tokens. Feel free to read  my previous blog post about [[Self attention]].
<!-- more -->

What are the kind of tokens we may not want a model to "attend to"? They're usually:

1. Padding tokens - These tokens are added in some architectures (such as BERT) such that all input token sequences are of the same length, and you do not want them to affect your final output.  
2. Future tokens - when training some decoder architectures, you want the "weighting" by the attention mechanism placed on the *value* vectors corresponding to tokens further along in the sequence in the training data to be zero. Because if it is *non-zero* your model is basically "cheating" during training by looking at what it is supposed to predict, and will not perform as well during inference time.

With this in mind, let's see how masking tends to be implemented in practice. 

# Masking - the basics
The main way to make sure the tokens you want to mask are not being attended to is by manipulating the inputs to the softmax function. As a reminder, the softmax function is defined as the following - given an array of inputs $[x_0, x_1, ... x_i, ..., x_N]$ the softmax function returns an array of $[p_0, p_1, ... p_i, ... p_N]$ where 

$$
p_i = \frac{e^{x_i}}{\sum^{N}_{i=1}{e^{x_i}}}
$$

and $\sum_ip_i = 1$. 

This softmax function gets applied on the output of the $k^Tq$ operation in the attention. The main "trick" when it comes to masking in a vectorized way is basically then overwriting this value with a large *negative* value. 

To see this in action, let us see how you'd make sure future tokens aren't being attended to in pure Python for loops. 

First, let's set up the toy example:

```python

import numpy as np

def softmax(items_in: list):
  e_x = np.exp(items_in - np.max(items_in))
  return e_x / e_x.sum()


# Set seed so we get the same random numbers
np.random.seed(3)
# Number of inputs
N = 4
# Number of dimensions of each input
D = 3

all_x = []
# Create elements x_n and append to list
for n in range(N):
  all_x.append(np.random.normal(size=(D,1)))

print("all_x: ")
print(all_x)
A_q = np.random.normal(size=(D,D))
A_k = np.random.normal(size=(D,D))
A_v = np.random.normal(size=(D,D))

# all the biases are of dimension Dx1
b_q = np.random.normal(size=(D,1))
b_k = np.random.normal(size=(D,1))
b_v = np.random.normal(size=(D,1))

all_queries = []
all_keys = []
all_values = []
# For every input
for x in all_x:
  query = A_q @ x + b_q
  key = A_k @ x + b_k 
  value = A_v @ x + b_v

  all_queries.append(query)
  all_keys.append(key)
  all_values.append(value)
```

In this above example, the columns of `all_x` are the `D` dimensioned data points, and there are `N=3` of them. This can be thought of as the output of the embedding layer in a standard transformer block.

Assume this is a causal mask we are trying to apply, i.e. for the `i_th` input, there should only be non-zero attention weights for *values* with an index `<=i`.
If that was confusing, let's illustrate this in code:

```python
all_outs = []

for i in range(N):
    print("=="*10)
    print(f"i: {i}")
    all_kj_qi = []
    q_i = all_queries[i]
    for j in range(i + 1):  # <- no future tokens will be attended to. If you want to attent to all tokens, this line would change to range(N) instead of range(i + 1) 
        key_j = all_keys[j]
        dot_product = np.dot(key_j.T, q_i).squeeze()
        all_kj_qi.append(dot_product)

    print("before softmax:")
    print(all_kj_qi)
    attention = softmax(all_kj_qi)
    print(f"attentions: {attention}")
    out_i = sum(attention[i] * all_values[i] for i in range(len(attention)))
    all_outs.append(out_i)
    print("=="*10)
```

The output would be: 
```text
====================
i: 0
before softmax:
[array(8.0985502)]
attentions: [1.]
====================
====================
i: 1
before softmax:
[array(1.33534917), array(1.04085729)]
attentions: [0.57309546 0.42690454]
====================
====================
i: 2
before softmax:
[array(2.95774823), array(0.4307023), array(2.98146278)]
attentions: [0.47530942 0.0379747  0.48671588]
====================
====================
i: 3
before softmax:
[array(7.39220366), array(0.99822195), array(6.05577164), array(5.18771535)]
attentions: [0.72739962 0.00121591 0.19114723 0.08023724]
====================
```

This basically means the following - 

$$
out_0 = 1.0 \times v_0
$$

$$
out_1 = 0.573 \times v_0 + 0.427 \times v_1
$$

$$
out_2 = 0.475 \times v_0 + 0.038 \times v_1 + 0.487 \times v_2
$$

and so forth. As you can see, $out_i$ only depends on $v_j$ if $j<=i$. 

So far, so good, but these pure Python `for` loops are slow, and you want to do this in a vectorized way. How would you do that?

As mentioned previously, you do this by "forcing" the pre-softmax co-efficients to be large negative values at the positions you want to mask, so that the post-softmax coefficients at these positions are 0. This is best illustrated in this snippet of code: 

```python
all_outs = []

for i in range(N):
    print("=="*10)
    print(f"i: {i}")
    all_kj_qi = [] # <-- will be a 1 x N vector
    q_i = all_queries[i]
    for j in range(N):
        key_j = all_keys[j]
        dot_product = np.dot(key_j.T, q_i).squeeze()
        all_kj_qi.append(dot_product)

    print("before adding:")
    print(all_kj_qi)
    to_add = np.array([0.] * (i + 1) + [-np.inf] * (N - i - 1))
    all_kj_qi += to_add
    print("after adding:")
    print(all_kj_qi)
    attention = softmax(all_kj_qi) # <-- 1 x N vector that sums to 1
    print(f"attentions: {attention}")
    out_i = sum(attention[i] * all_values[i] for i in range(N))
    all_outs.append(out_i)
    print("=="*10)
```

resulting in the output: 

```text
====================
i: 0
before adding:
[array(8.0985502), array(-1.41964676), array(0.40887655), array(-4.51894638)]
after adding:
[8.0985502      -inf      -inf      -inf]
attentions: [1. 0. 0. 0.]
====================
====================
i: 1
before adding:
[array(1.33534917), array(1.04085729), array(3.15550058), array(4.36428589)]
after adding:
[1.33534917 1.04085729       -inf       -inf]
attentions: [0.57309546 0.42690454 0.         0.        ]
====================
====================
i: 2
before adding:
[array(2.95774823), array(0.4307023), array(2.98146278), array(2.65666126)]
after adding:
[2.95774823 0.4307023  2.98146278       -inf]
attentions: [0.47530942 0.0379747  0.48671588 0.        ]
====================
====================
i: 3
before adding:
[array(7.39220366), array(0.99822195), array(6.05577164), array(5.18771535)]
after adding:
[7.39220366 0.99822195 6.05577164 5.18771535]
attentions: [0.72739962 0.00121591 0.19114723 0.08023724]
====================
```

You can see that the attention values are the same as the previous attention values! This second snippet is *much* simpler to write in a vectorized manner. First let's set everything up as matrices instead of vectors:

```python
X = np.array(all_x).squeeze()
Q = X @ A_q.T + b_q.T  # <- N x D matrix
K = X @ A_k.T + b_k.T  # <- N x D matrix
V = X @ A_v.T + b_v.T  # <- N x D matrix

# show that our set up above using pure python for loops, and this matrix set up are equivalent
assert (Q == np.array(all_queries).squeeze()).all()

def softmax_cols(data_in):
    # Exponentiate all of the values
    # keepdims=True IS VERY IMPORTANT
    _data_in = data_in - np.max(data_in, axis=1, keepdims=True)
    exp_values = np.exp(_data_in)
    # Sum over columns
    denom = np.sum(exp_values, axis=1, keepdims=True)
    # Compute softmax
    softmax = exp_values / denom
    # return the answer
    return softmax

```

Now, to look at what the pre-softmax values are, you can inspect what `Q @ K.T` is: 

```python
>> Q@K.T
array([[ 8.0985502 , -1.41964676,  0.40887655, -4.51894638],
       [ 1.33534917,  1.04085729,  3.15550058,  4.36428589],
       [ 2.95774823,  0.4307023 ,  2.98146278,  2.65666126],
       [ 7.39220366,  0.99822195,  6.05577164,  5.18771535]])
```

note that each row is basically the array that was printed out as part of the `before adding:` part in the previous output snippet! For illustration:

![[Pasted image 20240708164209.png]]

And now you need the attention mask: 

```python

>> to_add = np.triu(-np.inf * np.ones((N, N)), k=1)
>> to_add

array([[  0., -inf, -inf, -inf],
       [  0.,   0., -inf, -inf],
       [  0.,   0.,   0., -inf],
       [  0.,   0.,   0.,   0.]])
```

Then you'd add this mask to the $QK^T$ value. 
```python
>>pre_softmax = Q@K.T + to_add
>>pre_softmax
array([[8.0985502 ,       -inf,       -inf,       -inf],
       [1.33534917, 1.04085729,       -inf,       -inf],
       [2.95774823, 0.4307023 , 2.98146278,       -inf],
       [7.39220366, 0.99822195, 6.05577164, 5.18771535]])
```

Once done, just put it all through the softmax function!
```python
>>softmax_cols(pre_softmax)
array([[1.        , 0.        , 0.        , 0.        ],
       [0.57309546, 0.42690454, 0.        , 0.        ],
       [0.47530942, 0.0379747 , 0.48671588, 0.        ],
       [0.72739962, 0.00121591, 0.19114723, 0.08023724]])
```

This is your attention matrix! It is the same output we get as when we used the pure Python loops.

As a recap, in a vectorized way, all you need is just the following lines of code:

```python
Q = X @ A_q.T + b_q.T  # <- N x D matrix
K = X @ A_k.T + b_k.T  # <- N x D matrix
V = X @ A_v.T + b_v.T  # <- N x D matrix

to_add = np.triu(-np.inf * np.ones((N, N)), k=1)
pre_softmax = Q@K.T + to_add
attention = softmax_cols(pre_softmax)
out = attention @ V

assert (out == np.array(all_outs).squeeze()).all()  # <- shows equivalence
```

# Masking in the HF library

Now that you know the basics of how masking is supposed to work, let's take a look at how the Huggingface library implements this, specifically in the BERT modules. 

Let's set up this toy example: 

```python
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

words = [
	["word1*word2|word3"],
	["word1*", "word2|", "word3"],
]
tokenizer_out = tokenizer(
	text=words,
	is_split_into_words=True,
	return_offsets_mapping=True,
	return_overflowing_tokens=True,
	truncation="longest_first",
	padding="max_length",
	return_tensors="pt",
	max_length=512,
	stride=0,
	return_length=True,
)
# Input IDs and attention mask
input_ids = tokenizer_out["input_ids"]
attention_mask = tokenizer_out["attention_mask"]
```

If you look at what input_ids and the attention_mask are: 
```python
input_ids
tensor([[ 101, 2773, 2487, 1008, 2773, 2475, 1064, 2773, 2509,  102,    0, 0, ...],
        [ 101, 2773, 2487, 1008, 2773, 2475, 1064, 2773, 2509,  102,    0, 0, ...]])

attention_mask
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]])

```

The `1`s in the `attention_mask` are used to indicate tokens that *should* be attended to, while the `0`s are used to indicate tokens that you don't want the model to attend to.

Where does this attention mask get computed in the `BertTokenizerFast`? You have to chase down the `return_offsets_mapping`, and you will go all the way to the `_batch_encode_plus` method of the `PreTrainedTokenizerFast` class, which calls `self._tokenzier.encode_batch`. As part of the `fast` implementation of tokenizers, this method is implemented in Rust for performance reasons!

You can still follow the logic of where the attention mask is computed in the non-fast implementation called `BertTokenizer`. To do so, you'll have to chase down the functions in the `forward` method all the way till you reach the `_pad` method of the `PreTrainedTokenizerBase` class, where you can see how Huggingface deals with the various padding strategies available. In the snippet above, we chose `max_length`, the relevant parts of the code are: 

```python
# Initialize attention mask if not present.
	if return_attention_mask and "attention_mask" not in encoded_inputs:
		encoded_inputs["attention_mask"] = [1] * len(required_input)
	difference = max_length - len(required_input)
	encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
```

We next turn our attention to the `forward` method of the `BertModel` to see what is going on. You can go through the forward pass with -

```python
model = BertModel.from_pretrained(model_name)

# Inference with the mask
model.eval()
outputs = model(input_ids, attention_mask=attention_mask)
```

Once it enters the `forward` method, you'll see that there's a method called `get_extended_attention_mask` that is part of the `BertModel` (actually its inheritance is a bit more complicated - it is defined in the `ModuleUtilsMixin` and the chain of inheritance is something like `ModuleUtilsMixin` -> `PreTrainedModel` -> `BertPreTrainedModel` -> `BertModel`, where the `->` is shorthand for "is inherited by").

The conversion of this array of 0s and 1s to the form of `to_add` is done here in this line:

```python
extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
```

where the 1s in the `attention_mask` get converted to 0s and the `0`s get converted into a large negative value (like the `-np.inf` in the code snippets I provided). 

This extended mask then gets passed trough `BertEncoder`, to `BertAttention` to `BertSelfAttention` where it is finally used in the `forward` method: 

```python

class BertSelfAttention(nn.Module):
	...
	def forward(
		self,
		...,
		attention_mask,
		...,
	):
		# Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        ...
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        ...
        return outputs
```

The `...` is there to get rid of all the "noise" that clutters our understanding of how masking is being used in the underlying torch modules in the HF library on Bert. 

As you can see, the structure is very similar to the code snippets that I wrote from scratch above!

Hopefully, you now have more clarity about how this often overlooked (but important) part of any transformer block functions. 


