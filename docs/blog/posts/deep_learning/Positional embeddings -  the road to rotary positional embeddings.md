---
title: RoPE embeddings
draft: false
date: 2024-07-02
---
Rotary positional embeddings (RoPE) is yet another tool people use to improve the performance of BERT based models.

For a description of what embeddings are, and how they get used in attention/transformers, feel free to take a look at a previous post - [[Self attention#Embedding]] - for more information. The main point of an embedding module is to serve as a lookup table, mapping various kinds of integer ids - token ids, position ids etc. - into vectors that will then get used in the attention mechanism.

While *token* embeddings are one part of the puzzle, it is also important to inject the notion of a token's position in a sequence in the inputs to the attention mechanism. This injection is the subject of various methods in *positional* embeddings/encodings. 
<!-- more -->

Here we'll go through the following - 

1. Quickly recap the following:
	1. The encoding scheme of proposed in the original transformers paper
	2. The embedding scheme in the original BERT model
		1. How it is implemented in HuggingFace

2. Rotary positional embeddings (RoPE)
	1. Brief description behind wanting to use relative positional information
	2. How it is implemented in code

NOTE: This post, like all other posts is best viewed in dark mode.

## Sinusoidal embeddings from transformers paper
As a recap the positional encoding in the original transformers paper was:

$$
\text{PE}_{(pos, 2i)} = \text{sin}(\text{pos} \times \theta_{i})
$$

$$
\text{PE}_{(pos, 2i + 1)} = \text{cos}(\text{pos} \times \theta_{i})
$$

where 

$$
\theta_{i} = 10,000^{\frac{-2i}{d}}
$$

and $d$ is the dimension of the vector. 

To visualise these quantities, let's plot some of them and how they vary. Let's start with $\theta_i$ and how it varies with $i \in [0, d]$, with $d=64$.   

<details>
<summary>Plotting code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt


dim = 64

indices = np.arange(0, dim // 2)
base = 10_000
power_term = np.power(base, -2 * indices / dim)

plt.figure(figsize=(15,10))
plt.ylabel(r'$\mathrm{\theta_i = 10000^{-\frac{2i}{d}}}$')
plt.xlabel("i")
plt.plot(power_term)
plt.show()
```
</details>

![[Pasted image 20240730183240.png]]

Now, let's see how $\sin(m\theta_i)$ varies for different values of $m$, restricting ourselves to $m=100$. Recall that $m$ here represents the positions of tokens. 

<details>
<summary>Plotting code </summary>
```python
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle("Sine term for different values of m", fontsize=16)

m_values = [1, 10, 20, 99]  # Different values of m for each subplot

for i, ax in enumerate(axs.flat):
    m = m_values[i]
    ax.plot(np.sin(m * power_term))
    ax.set_title(f"m = {m}")
    ax.set_xlabel("i")
    ax.set_ylabel(r'$\mathrm{\sin(m \times 10000^{-\frac{2i}{d}})}$')

plt.tight_layout()
plt.show()

```
</details>


![[Pasted image 20240731120421.png]]
Each subplot here represents the positional encoding for a data input. For each subplot, the x-axis represents the $i^{th}$ position of the $d$ dimensional vector in position $m$. I show these plots because it is not intuitive what the shape of a sinusoid of a exponential term would be. 

With this in mind, let us see a 2D plot of this encoding scheme, using  `seq_len = 10` and `dim=64`. 

<details>
<summary>Plotting code</summary>
```python
def sinusoidal_embeddings(positions: np.array, dim: int, base=10000):
    """Interleaved sinusoidal position embeddings.

    Like in the original transformer paper - interleaved sin and cos.
    """
    indices = np.arange(0, dim // 2)
    power_term = np.power(base, -2 * indices / dim)
    angle = np.einsum("...,d->...d", positions, power_term)
    embeddings = np.stack([np.sin(angle), np.cos(angle)], axis=-1)
    embeddings = embeddings.reshape(list(embeddings.shape[:-2]) + [-1])
    return embeddings

seq_len = 10
positions = np.arange(seq_len)
embeddings = sinusoidal_embeddings(positions=positions, dim=dim)

plt.figure(figsize=(12,8))
plt.pcolormesh(embeddings, cmap='viridis')
plt.xlabel('Embedding Dimensions')
plt.xlim((0, dim))
plt.ylim((0, seq_len))
plt.ylabel('Token position')
plt.colorbar()
plt.show()
```
</details>

![[Pasted image 20240731131522.png]]

Each row represents the encoding that will be added to the embedding of a token.

But even this plot is not useful in visualising how this "helps" in injecting position information into the inputs. One visualisation that may help is by looking at the Euclidean distance of each of the vectors to a vector at a particular position index. The plot below can provide some intution (here `seq_len` is 100 again):

<details><summary>Plotting Code</summary>
```python
seq_len = 10
positions = np.arange(seq_len)
embeddings = sinusoidal_embeddings(positions=positions, dim=dim)

fig, axs = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle("Euclidean of all vectors to a vector at a particular position", fontsize=16)

vector_idxes = [0, 10, 20, 75]  # different position for each subplot

for i, ax in enumerate(axs.flat):
    idx = vector_idxes[i]
    distances = np.linalg.norm(embeddings - embeddings[idx], axis=1)
    ax.plot(np.arange(distances.shape[0]), distances, marker='o')
    ax.set_title(f'Distance of Vectors to the {idx}th Vector')
    ax.set_xlabel('Vector Index')
    ax.set_ylabel('Euclidean Distance')
    ax.grid(True)

plt.tight_layout()
plt.show()
```
</details>

![[Pasted image 20240731133350.png]]

Here you can see that these graphs have a "V" shape - for a given vector, vectors that are close to it positionally *generally* have a smaller Euclidean distance than those that are further away positionally. 

You can also look at the "frequency" of the sinusoids at each $i$ as $m$ varies:

<details><summary>Plotting Code</summary>
```python
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle("Frequency variation for each i as m varies", fontsize=16)

dim_idxes = [1, 10, 20, 30]  # Different values of m for each subplot

for i, ax in enumerate(axs.flat):
    dim_idx = dim_idxes[i]
    ax.plot(embeddings[:, dim_idx])
    ax.set_title(f'i = {dim_idx}')
    ax.set_xlabel('m')
    ax.set_ylabel(r'$\mathrm{\theta_i = 10000^{-\frac{2i}{d}}}$')
    ax.grid(True)

plt.tight_layout()
plt.show()
```
</details>


![[Pasted image 20240731143104.png]]

This makes sense because the frequency of the sinusoid is $\theta_i$. $\theta_i$ has a higher value for when $i$ is smaller, and a smaller value when $i$ is larger. 

### What are the limitations of of the sinusoidal scheme proposed in the original transformers paper? 

The most obvious issues are as listed:
1. The main problem with the scheme presented above is that everything is fixed. None of the components - the variation in the frequency range, the sinusoidal function itself etc. - gets updated during training. 
2. The different sinusoids, while expressive, may not be the correct kind of expressive for all sequences and may not be capturing the complex positional relationships in the data.
3. For longer sequence lengths, you can see that the difference blurs for vectors that are further away. 

## BERT embeddings

The [BERT paper](https://arxiv.org/pdf/1810.04805) uses learned positional embeddings (though the paper doesn't explicitly state it). The main idea is this - instead of using *predefined sinusoids* like the original transformers paper used, the BERT paper defers the learning of how best to deal with a token's position to the model itself, so that the model learns this in conjunction with the other weights. 

Yet another embedding layer is defined, and instead of serving as a look up table for token ids to $d$ dimensional vectors, this embedding layer serves as a look up table for *a token's position in the sequence* to a $d$ dimensional vector.

That is it, there's nothing more complicated. Let's quickly look at how the HuggingFace library does it.

### BERT positional embeddings in the HuggingFace library

Let's start with the `BertEmbeddings` class to see what happens.
In the `BertEmbeddings` class, the `self.position_ids` is basically a 1 dimensional tensor that goes from `0` to `max_tokens - 1`. In the default case, `max_tokens` is 512, so `self.position_ids` is a $1 \times 512$  `tensor([[ 0, 1, 2, 3, ... 511]])` (in code, this happens [here](https://gitkraken.dev/link/dnNjb2RlOi8vZWFtb2Rpby5naXRsZW5zL2xpbmsvci80M2JhZGYyMTdkMWNjZmFmNDg2ZTJjYmIxYjM1NjcyMjZiNWU5NWJmL2Yvc3JjL3RyYW5zZm9ybWVycy9tb2RlbHMvYmVydC9tb2RlbGluZ19iZXJ0LnB5P3VybD1odHRwcyUzQSUyRiUyRmdpdGh1Yi5jb20lMkZodWdnaW5nZmFjZSUyRnRyYW5zZm9ybWVycy5naXQ%3D?origin=gitlens)). Remember that for BERT, the number of input tokens is fixed, and shorter token sequences are padded. 
In the `forward` method of `BertEmbeddings`, you can see that this tensor is then passed through a `torch.nn.Embedding` module, which is nothing more than a look up table casting ints to vectors. The result is then added to the token embeddings.

```python

class BertEmbeddings(nn.Module):
	def __init__(self, config):
		...
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
		...
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		
		# same as self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
		self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        
		...

	def forward(self, input_ids, ...):
		...
		inputs_embeds = self.word_embeddings(input_ids)
		...
		position_embeddings = self.position_embeddings(position_ids)
		
		embeddings += position_embeddings  # <- main line here
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
		
```


That is basically how embedding is done in `BertEmbeddings`. 
As an aside, let's also quickly look at how the output of this module is used. Looking at the `forward` method of the `BertModel` in Huggingface, you'll see that these embedding outputs become the `hidden_state`:

In `BertModel`:

```python

def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        ...
        encoder_outputs = self.encoder(
                embedding_output,  # <- this is "hidden_state" for the encoder
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
```

and then you go into the `BertEncoder`: 
```python
def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        ...
        layer_outputs = layer_module(
                hidden_states,  # <- this contains the embeddings
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
        )
```

and now go all the way into the `BertSelfAttention` and investigate it line by line:


```python
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        ...
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)


        def forward(
                self,
                hidden_states: torch.Tensor,  # <- this contains the embeddings, used as input to the attention mechanism
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
                mixed_query_layer = self.transpose_for_scores(self.query(hidden_states))
                ...
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

```

## RoPE embeddings

### Background
In the embedding/encoding schemes above, the model has to learn the absolute position embeddings. But this is often not very useful under the following circumstances:

1. Often times when training on some datasets, shorter unrelated token sequences are stacked together and trained on together. Here absolute positional embeddings make no sense because the actual first token of a sentence would have some arbitrary positional information based on what came before it.
2. Other times, long token sequences are broken up into shorter sequences before passing through the attention layers. In this scenario absolute positions don't correspond to the actual positions of the tokens in the original sequences.
3. In the standard attention mechanism, the dot product no longer cares about the positions of tokens. This means the attention mechanism is only ever *indirectly* dependent on the position of tokens because the positional embeddings are added to the token embeddings *before* the matrix multiplication.

![[Pasted image 20240715130105.png]]



For such cases, we want an embedding scheme that efficiently does some kind of relative positional embedding, and also explicitly "injects" this knowledge into the attention mechanism. What does relative positional embedding mean? We basically want a function $f(\mathbf{x}, l)$ that takes in as arguments an input token sequence $\mathbf{x}$ and its position $l$ such that the *dot product* between 2 vectors $\mathbf{x}_1$ and $\mathbf{x}_2$ in positions $l_1$ and $l_2$ is only sensitive to $\mathbf{x}_1$ and $\mathbf{x}_2$, and the relative position $l_1 - l_2$. 

There are many ways to do this, but the scheme that most of the field has settled on was first proposed in the [RoFormer paper](https://arxiv.org/pdf/2104.09864), which we talk about in the next section.

### What are RoPE embeddings?

The RoFormer paper, and the [EleutherAI blogpost](https://blog.eleuther.ai/rotary-embeddings/) that the paper mentions, explain the intuition behind rotational positional embeddings in detail. But both contains a lot of detail that obfuscated the essence (the "[Visual Intuition](https://blog.eleuther.ai/rotary-embeddings/#visual-intuition)" section of the blogpost continues to confuse me) for me.

If you have found yourself on the same boat, below is hopefully a simpler and more practical explanation of it.

The easiest way to to preserve the notion of some kind of "relative" distance between two token embeddings in the pre-softmax attention step, is to make use of the angle between them in their $d$ dimensional space. This is because the dot product of two vectors is proportional to the cosine of the angle between them, and the pre-softmax matrix multiplication step is nothing but a series of dot products. 

A cunning way to do so would be to multiply the token embedding in the $m^{th}$ position in the sequence by the following rotational matrix (using 1 as the first index for clarity + ease of using latex):


$$
\mathbf{R}_{d, m} =
\tiny{\begin{bmatrix}
\cos(m\theta_1) & - \sin(m\theta_1) & 0 & 0 & ...  & 0 & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & ...  & 0 & 0 \\
0 & 0 & \cos(m\theta_2) & - \sin(m\theta_2) & ...  & 0 & 0 \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & ...  & 0 & 0 \\
... \\
0 & 0 & 0 & 0 & ...  & \cos(m\theta_{d/2}) & - \sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & ...  & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2}) \\
\end{bmatrix}}
$$

where 


$$
\theta_i = 10,000 ^{\frac{-2i}{d}}
$$

i.e. the original definition of the angle as in the transformer paper.

So now, your key values become: 

$$
\mathbf{k}'_{m} = \mathbf{R}_{d, m}\mathbf{k}_{m}
$$

and your query values become 

$$
\mathbf{q}'_{n} = \mathbf{R}_{d, n}\mathbf{q}_{n}
$$

These are now inputs to your attention mechanism. To see why multiplying your keys and queries with this rotational matrix is cunning, let's see what happens when you take the dot product of a key vector in position $m$ and a query vector in position $n$ (as is usual in the attention mechanism):

$$
\mathbf{k'}^{T}_{m}\mathbf{q'}_{n} = \mathbf{k}^T_{m}\mathbf{R}^T_{d, m}\mathbf{R}_{d, n}\mathbf{q}_{n}
$$

The quantity $\mathbf{R}^T_{d, m}\mathbf{R}_{d, n}$ is nothing but another rotation matrix! You can see the derivation pretty simply:

$$
\mathbf{R}^T_{d, m}\mathbf{R}_{d, n} = 
$$



$$
\tiny{
\begin{bmatrix}
\cos(m\theta_1) &  \sin(m\theta_1) & 0 & 0 & ...  & 0 & 0 \\
- \sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & ...  & 0 & 0 \\
0 & 0 & \cos(m\theta_2) &  \sin(m\theta_2) & ...  & 0 & 0 \\
0 & 0 & -\sin(m\theta_2) & \cos(m\theta_2) & ...  & 0 & 0 \\
... \\
0 & 0 & 0 & 0 & ...  & \cos(m\theta_{d/2}) &  \sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & ...  & -\sin(m\theta_{d/2}) & \cos(m\theta_{d/2}) \\
\end{bmatrix}
\begin{bmatrix}
\cos(n\theta_1) & - \sin(n\theta_1) & 0 & 0 & ...  & 0 & 0 \\
\sin(n\theta_1) & \cos(n\theta_1) & 0 & 0 & ...  & 0 & 0 \\
0 & 0 & \cos(n\theta_2) & - \sin(n\theta_2) & ...  & 0 & 0 \\
0 & 0 & \sin(n\theta_2) & \cos(n\theta_2) & ...  & 0 & 0 \\
... \\
0 & 0 & 0 & 0 & ...  & \cos(n\theta_{d/2}) & - \sin(n\theta_{d/2}) \\
0 & 0 & 0 & 0 & ...  & \sin(n\theta_{d/2}) & \cos(n\theta_{d/2}) \\
\end{bmatrix}
}
$$

$$
= \tiny{\begin{bmatrix}
\cos(m\theta_1)\cos(n\theta_1) + \sin(m\theta_1)\sin(n\theta_1) &  -\cos(m\theta_1)\sin(n\theta_1) + \sin(m\theta_1)\cos(n\theta_1) & 0 & 0 & ...  & 0 & 0 \\
- \sin(m\theta_1)\cos(n\theta_1) + \cos(m\theta_1)\sin(n\theta_1) & \cos(m\theta_1)\cos(n\theta_1) + \sin(m\theta_1)\sin(n\theta_1) & 0 & 0 & ...  & 0 & 0 \\
0 & 0 & ... &  ... & ...  & 0 & 0 \\
0 & 0 & ... & ... & ...  & 0 & 0 \\
... \\
0 & 0 & 0 & 0 & ...  & \cos(m\theta_{d/2})\cos(n\theta_{d/2}) + \sin(m\theta_{d/2})\sin(n\theta_{d/2}) &  -\cos(m\theta_{d/2})\sin(n\theta_{d/2}) + \sin(m\theta_{d/2})\cos(n\theta_{d/2})\\
0 & 0 & 0 & 0 & ...  & - \sin(m\theta_{d/2})\cos(n\theta_{d/2}) + \cos(m\theta_{d/2})\sin(n\theta_{d/2}) & \cos(m\theta_{d/2})\cos(n\theta_{d/2}) + \sin(m\theta_{d/2})\sin(n\theta_{d/2})\\
\end{bmatrix}
}
$$

$$
= \tiny{\begin{bmatrix}
\cos(n-m)\theta_1 & - \sin(n-m)\theta_1 & 0 & 0 & ...  & 0 & 0 \\
\sin(n-m)\theta_1 & \cos(n-m)\theta_1 & 0 & 0 & ...  & 0 & 0 \\
0 & 0 & \cos(n-m)\theta_2 & - \sin(n-m)\theta_2 & ...  & 0 & 0 \\
0 & 0 & \sin(n-m)\theta_2 & \cos(n-m)\theta_2 & ...  & 0 & 0 \\
... \\
0 & 0 & 0 & 0 & ...  & \cos(n-m)\theta_{d/2} & - \sin(n-m)\theta_{d/2} \\
0 & 0 & 0 & 0 & ...  & \sin(n-m)\theta_{d/2} & \cos(n-m)\theta_{d/2} \\
\end{bmatrix}
}
$$

$$
= \mathbf{R}_{d, n-m}
$$

So we now basically have a way of "injecting" the relative distance between 2 tokens in a sequence into the attention mechanism!  All this theory still does not completely explain exactly how this leads to better training, as is common with all transformers behaviour. But the common consensus in the field has been that this injection, and corresponding multiplicative and sinusoidal relationship between relative position and the learned weights, makes it "easier" (more data efficient) for the model to learn sequences - a consensus that has been proved empirically over and over again.

Now, let's see some basic plots of this. 

First, here's the code to generate $\mathbf{R}_{d, m}$ 

```python
def get_rot_matrix(dim: int, m: int):
    # same as 
    # indices = np.arange(0, dim // 2)
    # base = 10_000
    # theta = np.power(base, -2 * indices / dim)
    theta = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    theta = m * theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    main_diagonal = np.repeat(cos_theta, 2)
    rot = np.diag(main_diagonal)
    off_diagonal = np.zeros(dim)
    off_diagonal[::2] = sin_theta
    rot += np.diag(-off_diagonal[:-1], k=1)
    rot += np.diag(off_diagonal[:-1], k=-1)
    return rot
```

To see how this affects the pre-softmax layer, let's take 2 identity vectors of dimension $d=64$,

```python
dim = 64
k = np.ones((dim, 1))
q = np.ones((dim, 1))
```

Without the rotation matrix, the value of $\mathbf{k}_m^T\mathbf{q}_n$ has a constant value of 64 regardless of the value of $m$ and $n$. With the rotational matrix, lets's see how the value of $\mathbf{k}^T_{m}\mathbf{R}_{d, n-m}\mathbf{q}_{n}$ varies with $n-m$ with the following plot:

<details>
<summary>Plotting code</summary>
```python
x = np.arange(0, 100)
plt.plot(x, [(k.T@get_rot_matrix(dim=dim, m=_m)@q).item() for _m in x])
plt.xlabel(r'Relative distance $\mathrm{n - m}$')
plt.ylabel(r'$\mathbf{k}^T_{m}\mathbf{R}_{d, n-m}\mathbf{q}_{n}$')
plt.axhline(y=64, color='r', linestyle=':')
plt.text(x=70, y=62, s="y=64", color='r', fontsize=12)
```
</details>

![[Pasted image 20240803111003.png]]

This by itself doesn't actually show the mechanics of what will happen during training. During training, the vectors $\mathbf{k}_m$ and $\mathbf{q}_n$ are themselves the result of multiplying a weight vector $\mathbf{W}_k$ (for the key values), $\mathbf{W}_q$ (for the query values) to the token embedding at positions $m$ and $n$ respectively. This means that the pre-softmax value is actually 

$$
\mathbf{k}^T_{m}\mathbf{R}_{d, n-m}\mathbf{q}_{n} = \mathbf{x}_m^T\mathbf{W}_k^T\mathbf{R}_{d, n-m}\mathbf{W}_q\mathbf{x}_n
$$

Where $\mathbf{W}_k, \mathbf{W}_q$ are learned weights. This means that the exact nature of the sinusoidal patterns we see in the plot as $n-m$ increases can get very finely tuned during the training process.

### Implementation in practice

So far, the code we have looked at is at a vector level. Now let's see how this would get implemented when you have to do it for an entire sequence of tokens. 

The query matrix $\mathbf{Q} \in \mathbb{R}^{N \times d}$, where $N$ is the number of tokens in the sequence and $d$ is the dimension of each data point, can be represented as (switching back to 0th index because we're going to convert everything to code):
 
$$
\mathbf{Q} = \left[
\begin{array}{c}
\hphantom{-}--- q_0^T--- \hphantom{-} \\
\hphantom{-}--- q_1^T--- \hphantom{-} \\
\hphantom{-}--- q_2^T--- \hphantom{-} \\
...\\
\hphantom{-}--- q_{N-1}^T--- \hphantom{-} \\
\end{array}
\right]
$$

where $q_i \in \mathbb{R}^d, q_i = \mathbf{W}_q\mathbf{x}_i$ represents the query vector of the embedding of the token in the $i^{th}$ position $\mathbf{x}_i$.

Similarly for the key values:

$$
\mathbf{K} = \left[
\begin{array}{c}
\hphantom{-}--- k_0^T --- \hphantom{-} \\
\hphantom{-}--- k_1^T --- \hphantom{-} \\
\hphantom{-}--- k_2^T --- \hphantom{-} \\
...\\
\hphantom{-}--- k_{N-1}^T --- \hphantom{-} \\
\end{array}
\right]
$$

Let's focus only on the query matrix for now. After applying the rotational matrix to each of the vectors, let's call the result $\mathbf{Q'}$:

$$
\mathbf{Q'} = \left[
\begin{array}{c}
\hphantom{-}--- q_0^{,T}--- \hphantom{-} \\
\hphantom{-}--- q_1^{,T}--- \hphantom{-} \\
\hphantom{-}--- q_2^{,T}--- \hphantom{-} \\
...\\
\hphantom{-}--- q_{N-1}^{,T}--- \hphantom{-} \\
\end{array}
\right]
$$

$$
= \left[
\begin{array}{c}
\hphantom{-}--- \mathbf{R}^T_{d, 0}q_0^T--- \hphantom{-} \\
\hphantom{-}--- \mathbf{R}^T_{d, 1}q_1^T--- \hphantom{-} \\
\hphantom{-}--- \mathbf{R}^T_{d, 2}q_2^T--- \hphantom{-} \\
...\\
\hphantom{-}--- \mathbf{R}^T_{d, N-1}q_{N-1}^T--- \hphantom{-} \\
\end{array}
\right]
$$

The "naive" way to get the rotary positional embeddings of the key/query matrix is simply - 

1. Multiplying each row $\mathbf{q}_i$ with the rotational matrix $\mathbf{R}_{d, i}$. 
2. Setting the result of the previous step as the $i^{th}$ row of the resultant matrix.

In naive python, the code would be like this:

```python
np.random.seed(3)
Q = np.random.randn(5, 4)
Q_prime = np.zeros_like(Q)
for pos, q_i in enumerate(Q):
    rot = get_rot_matrix(Q_prime.shape[-1], pos)
    Q_prime[pos] = rot @ q_i.T
Q_prime
```

and the result you get is
```
array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
       [ 0.1486459 , -0.42509122, -0.07646744, -0.62779673],
       [ 0.45216792,  0.15874903, -1.33129326,  0.85816992],
       [-1.11375321, -1.5680929 ,  0.06214963, -0.40299454],
       [-0.81390684,  1.4235748 ,  1.02561261, -1.06090267]])
```

But there's a more cunning way of doing everything in a vectorised way and getting rid of the loop in the code snippet. Let's take a closer look at the multiplication of the rotational matrix with a query vector:

$$
\mathbf{R}_{d, m}\mathbf{q}_m =
\tiny{\begin{bmatrix}
\cos(m\theta_0) & - \sin(m\theta_0) & 0 & 0 & ...  & 0 & 0 \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & ...  & 0 & 0 \\
0 & 0 & \cos(m\theta_1) & - \sin(m\theta_1) & ...  & 0 & 0 \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & ...  & 0 & 0 \\
... \\
0 & 0 & 0 & 0 & ...  & \cos(m\theta_{\frac{d}{2} -1}) & - \sin(m\theta_{\frac{d}{2} -1}) \\
0 & 0 & 0 & 0 & ...  & \sin(m\theta_{\frac{d}{2} -1}) & \cos(m\theta_{\frac{d}{2} -1}) \\
\end{bmatrix}
\begin{bmatrix}
q^m_1 \\
q^m_2 \\
... \\
q^m_{d-1} \\
q^m_d \\
\end{bmatrix}
}
$$

(let's drop the $m$ superscript showing the sequence position on the individual elements of the $\mathbf{q}$ vector)

$$
=\tiny{
\begin{bmatrix}
\cos({m\theta_0})q_0 - \sin({m\theta_0})q_1 \\
\sin({m\theta_0})q_0 + \cos({m\theta_0})q_1 \\
\cos({m\theta_1})q_2 - \sin({m\theta_1})q_3 \\
\sin({m\theta_1})q_2 + \cos({m\theta_1})q_3 \\
... \\
\cos({m\theta_{\frac{d}{2} -1}})q_{d-1} - \sin({m\theta_{\frac{d}{2} -1}})q_d \\
\sin({m\theta_{\frac{d}{2} -1}})q_{d-1} + \cos({m\theta_{\frac{d}{2} -1}})q_d \\
\end{bmatrix}
}
$$
 
$$
= \tiny{
\begin{bmatrix}
\cos({m\theta_0}) \\ \cos({m\theta_0}) \\ \cos({m\theta_1}) \\ \cos({m\theta_1}) \\  ... \\ \cos({m\theta_{\frac{d}{2} -1}}) \\ \cos({m\theta_{\frac{d}{2} -1}})
\end{bmatrix}
\odot
\begin{bmatrix}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
... \\
q_{d-2} \\
q_{d-1} \\
\end{bmatrix}
}
$$
 
$$
+
$$
 
$$ 
\tiny{
\begin{bmatrix}
\sin({m\theta_0}) \\ \sin({m\theta_0}) \\ \sin({m\theta_1}) \\ \sin({m\theta_1}) \\  ... \\ \sin({m\theta_{\frac{d}{2} -1}}) \\ \sin({m\theta_{\frac{d}{2} -1}})
\end{bmatrix}
\odot
\begin{bmatrix}
- q_1 \\
q_0 \\
- q_3 \\
q_2 \\
... \\
- q_{d - 1} \\
q_{d - 2} \\
\end{bmatrix}
}
$$

where $\odot$ is the element-wise multiplication/Hadamard product.

So now, 

$$
\mathbf{Q'} = \left[
\begin{array}{c}
\hphantom{-}--- q_0^{,T}--- \hphantom{-} \\
\hphantom{-}--- q_1^{,T}--- \hphantom{-} \\
\hphantom{-}--- q_2^{,T}--- \hphantom{-} \\
...\\
\hphantom{-}--- q_{N-1}^{,T}--- \hphantom{-} \\
\end{array}
\right]
$$
 
$$
= \tiny{
\begin{bmatrix}
\cos({0\theta_0}) & \cos({0\theta_0}) & \cos({0\theta_1}) & \cos({0\theta_1}) &  ... & \cos({0\theta_{\frac{d}{2} -1}}) & \cos({0\theta_{\frac{d}{2} -1}}) \\
\cos({1\theta_0}) & \cos({1\theta_0}) & \cos({1\theta_1}) & \cos({1\theta_1}) &  ... & \cos({1\theta_{\frac{d}{2} -1}}) & \cos({1\theta_{\frac{d}{2} -1}}) \\
. \\
.\\
. \\
\cos({(N-1)\theta_1}) & \cos({(N-1)\theta_1}) & \cos({(N-1)\theta_2}) & \cos({(N-1)\theta_2}) &  ... & \cos({(N-1)\theta_{\frac{d}{2} -1}}) & \cos({(N-1)\theta_{\frac{d}{2} -1}}) \\
\end{bmatrix} \odot 
\left[
\begin{array}{c}
\hphantom{-}--- q_0^{,T}--- \hphantom{-} \\
\hphantom{-}--- q_1^{,T}--- \hphantom{-} \\
\hphantom{-}--- q_2^{,T}--- \hphantom{-} \\
...\\
\hphantom{-}--- q_{N-1}^{,T}--- \hphantom{-} \\
\end{array}
\right]
}
$$
 
$$
+
$$
 
$$
\tiny{
\begin{bmatrix}
\sin({0\theta_0}) & \sin({0\theta_0}) & \sin({0\theta_1}) & \sin({0\theta_1}) &  ... & \sin({0\theta_{\frac{d}{2} -1}}) & \sin({0\theta_{\frac{d}{2} -1}}) \\
\sin({1\theta_0}) & \sin({1\theta_0}) & \sin({1\theta_1}) & \sin({1\theta_1}) &  ... & \sin({1\theta_{\frac{d}{2} -1}}) & \sin({1\theta_{\frac{d}{2} -1}}) \\
. \\
.\\
. \\
\sin({(N-1)\theta_0}) & \sin({(N-1)\theta_0}) & \sin({(N-1)\theta_1}) & \sin({(N-1)\theta_1}) &  ... & \sin({(N-1)\theta_{\frac{d}{2} -1}}) & \sin({(N-1)\theta_{\frac{d}{2} -1}}) \\
\end{bmatrix}
\odot 
\left[
\begin{array}{c}
\hphantom{-}--- \text{rearranged-q}_0^{,T}--- \hphantom{-} \\
\hphantom{-}--- \text{rearranged-q}_1^{,T}--- \hphantom{-} \\
\hphantom{-}--- \text{rearranged-q}_2^{,T}--- \hphantom{-} \\
...\\
\hphantom{-}--- \text{rearranged-q}_{N-1}^{,T}--- \hphantom{-} \\
\end{array}
\right]
}
$$

where $\text{rearranged-q}_i$ is 

$$
\begin{bmatrix}
- q^i_1 \\
q^i_0 \\
- q^i_3 \\
q^i_2 \\
... \\
- q^i_{d-1} \\
q^i_{d - 2} \\
\end{bmatrix}
$$

So now, you can see pretty quickly that the way to implement rotary positional embeddings in a vectorised way is going to be something similar to:

```python
from einops import repeat, rearrange
import numpy as np


def rotate_every_two(x):
	"""
	Similar to what EleutherAI's implementation of the MeshTransformer is in JAX
	"""
    x1 = x[:, ::2]
    x2 = x[:, 1::2]
    x = np.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(x: np.array, seq_idx=1, dim_idx=-1) -> np.array:
    """apply rotary positional embedding

    x in an input array, and the assumption that the shape is
    (batch, seq_len, ..., dim)

    Args:
        x: the input array
        seq_idx: the index in x.shape that shows the sequence length.
            By default we assume it is 1
        dim_idx: the index in x.shape that shows the number of dimensions
            By default we assume it is -1

    Returns:
        a numpy array with rotary positional embedding applied
    """
    dim = x.shape[dim_idx]
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

    sinusoid_inp = np.einsum("i , j -> i j", np.arange(x.shape[seq_idx]), inv_freq)
    sines = np.sin(sinusoid_inp)
    cosines = np.cos(sinusoid_inp)
    
    def _apply_repeat(_arr: np.array) -> np.array:
        return repeat(_arr, "b n -> b (n j)", j=2)
    sin = _apply_repeat(sines)
    cos = _apply_repeat(cosines)
    return (x * cos) + (rotate_every_two(x) * sin)

apply_rotary_pos_emb(Q, seq_idx=0)
```

and the result is something like

```txt
array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
       [ 0.1486459 , -0.42509122, -0.07646744, -0.62779673],
       [ 0.45216792,  0.15874903, -1.33129326,  0.85816992],
       [-1.11375321, -1.5680929 ,  0.06214963, -0.40299454],
       [-0.81390684,  1.4235748 ,  1.02561261, -1.06090267]])
```

which is the same result you get using the "naive" python approach! This approach is also how the HuggingFace repository and other repositories implement rotary positional embeddings.

However, due to computational efficiency reasons, you'll see the `rotate_every_two` function not implemented in that interleaved way. For example in the [HuggingFace code for Llama](https://github.com/huggingface/transformers/blob/1749841a0e9d803984985e08e4df177ac5a8b1a9/src/transformers/models/llama/modeling_llama.py#L179-L183), you'll see this function instead:

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
So instead of interleaving the (negative) even and odd dimensional elements of each input tensor as is the case in $\text{rearranged-q}_i$, they just take the last half of the tensor, negate it, and concatenate it with the first half. So in this case $\text{rearranged-q}_i$ is 

$$
\begin{bmatrix}
- q_{\frac{d}{2}} \\
- q_{\frac{d}{2} + 1} \\
- q_{\frac{d}{2} + 2} \\
... \\
- q_{d - 1} \\
q_0 \\
q_1 \\
... \\
q_{\frac{d}{2} - 1} \\
\end{bmatrix}
$$

While it doesn't do the exact same thing, the main aim - that $\mathbf{k'}^{T}_{m}\mathbf{q'}_{n}$ is dependent on $m - n$ is still achieved. You can see that this is indeed the case by substituting $\text{rearranged-q}_n$ and $\text{rearranged-k}_n$ back into $\mathbf{q'}_{n}, \mathbf{k'}^{T}_{m}$ respectively, and then writing out the multiplication. To see that this is the case.

Hopefully this has been a useful discourse in rotary positional embeddings!