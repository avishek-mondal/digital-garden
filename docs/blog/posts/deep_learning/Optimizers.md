---
title: Implementing some optimizers from scratch
draft: true
date: 2025-03-27
---

Gradient based optimizations are fairly standard in the world of deep learning (an also for non-deep learning methods like LightGBM). When writing `torch` code, the most common pattern one encounters optimizers is in lines similar to the following: 

<!-- more -->

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

In this post, I will try and implement some optimizers from scratch to understand their properties better.

The 3 optimizers I will try out are 
1. (good old) SGD
2. Adam
3. AdamW

## SGD

## Adam

## AdamW