---
title: Various encoder and decoder implementations
draft: true
date: 2024-06-13
---
## How does masking work in BERT

There are many ways to 
Say you have a phrase, and you pass it in through BERT, like this: 

`word1*word2|word3` ->  BERT forward pass -> Some matrix. 

What are the steps you need to take in the BERT forward pass? This section will walk you through what happens using the HuggingFace implementation of BERT. 

There is a difference of what happens during training and what happens during inference. 



```python
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
We next turn our attention to the `forward` method of the BertModel to see what is going on. 



Then, looking at the `forward` method of the `BertModel` in Huggingface, 
