---
title: Notes on information theory for meta-learning
date: 2025-02-06
draft: false
---

When reading the "[Meta-Learning without Memorization](https://arxiv.org/pdf/1912.03820)" paper, there were several derivations that caught me off-guard. So, I'm going to put some basic derivations to paper to refer to in the future that will make reading papers easier. 

<!-- more -->

The main aim of this post is simple. The paper presents this plate diagram for a bottleneck variable: 

![[Pasted image 20250207142509.png]]

From this plate diagram, the authors present equation 2 when deriving a modified training objective to prevent memorization by regularizing the latent "activation" variable $z$: 

$$
\newcommand{\y}{\hat{y}^*}
\newcommand{\z}{z^*}
\newcommand{\x}{x^*}
\newcommand{\D}{\mathcal{D}}
\newcommand{\T}{\mathcal{T}}
\newcommand{\xx}{\mathbf{x}}
\newcommand{\yy}{\mathbf{y}}
\begin{aligned}
&I(\hat{y}^*;\mathcal{D}|z^*,\theta) \\
&\geq I(x^*;\hat{y}^*|\theta,z^*) \\
&= I(x^*;\hat{y}^*|\theta) - I(x^*;z^*|\theta) + I(x^*;z^*|\hat{y}^*,\theta) \\
&\geq I(x^*;\hat{y}^*|\theta) - I(x^*;z^*|\theta) \\
&= I(x^*;\hat{y}^*|\theta) - \mathbb{E}_{p(x^*)q(z^*|x^*,\theta)}\left[\log\frac{q(z^*|x^*,\theta)}{q(z^*|\theta)}\right] \\
&\geq I(x^*;\hat{y}^*|\theta) - \mathbb{E}\left[\log\frac{q(z^*|x^*,\theta)}{r(z^*)}\right] = I(x^*;\hat{y}^*|\theta) - \mathbb{E}[D_{KL}(q(z^*|x^*,\theta)||r(z^*))]
\end{aligned}
$$

The jumps that these equations take is not obvious to me at all. The aim of this post is to understand the first 3 steps of this equation. 

As a recap, the meta-learning setup is the following according to the paper: We assume tasks $\T_i$ are sampled from a distribution of tasks $p(\T)$ - for example the Omniglot dataset. For each task during **meta-training**, the model sees a set of training data $\D_i = (\xx_i, \yy_i)$ and a set of test data $D^*_i = (\xx^*_i, \yy^*_i)$ where $\xx_i, \yy_i, \xx^*_i, \yy^*_i$ are sampled from $p(x, y|\T_i)$. The model's prediction at test time for the task is denoted as $\y$. 

Obviously, don't show the rest of the article below to mathematicians, they'll probably nitpick and be generally miserable like they all are. I have not been too pedantic about notation, but hopefully I have been clear in showing what I want to show.

## Background

Let's cover a few background maths information we need.
### Jensen's inequality

The first is Jensen's inequality. But even before we get to that, let's show some basic things. 

#### Definition of convex functions 

Basically, a convex function are functions like the one below: 

![[Jensen's inequality drawing 1.png]]

In these functions, a line passing through two points of the function will be above the graph of the function itself. 

To be more rigorous about this, let's consider a function $f(x)$, and 2 points $x_1, x_2$. Let's now consider a third point $x'$ somewhere between $x_1, x2$. We can arbitrarily represent $x' := x_1 + \alpha(x_2 - x_1)$ for some $a \in [0, 1]$ - basically, if $\alpha = 0 \implies x' = x_1, \alpha = 1 \implies x' = x_2$. 

In the diagram above $y_{\text{curve}} = f(x') = f(x_1 + \alpha(x_2 - x_1)) = f((1 -\alpha)x_1 + \alpha x_2)$.

And finally, 

$$
\frac{y_{\text{line}} - f(x_1)}{x' - x_1} =  \frac{f(x_2) - f(x_1)}{x_2 - x_1}
$$

$$
\implies y_{\text{line}} = \frac{f(x_2) - f(x_1)}{x_2 - x_1}(x' - x_1) + f(x_1)
$$

$$
= \frac{f(x_2) - f(x_1)}{x_2 - x_1}(x_1 + \alpha(x_2 - x_1) - x_1) + f(x_1)
$$

$$
= \alpha(f(x_2) - f(x_1)) + f(x_1) = (1 - \alpha)f(x_1) + f(x_2)
$$

The definition of a convex function is basically $y_{\text{line}} >= y_{\text{curve}}, \forall \alpha \in [0, 1]$

$$
\implies (1 - \alpha)f(x_1) + f(x_2) \geq f((1 -\alpha)x_1 + \alpha x_2)
$$

The $\geq$ is switched to $\leq$ for concave functions. Examples of *concave* functions are log functions, while examples of convex functions are exponential functions. 

####  Extension to expected value of convex functions

Following the definition of a convex function, we now make the following statement: 
Given a convex function $f$, and $a_1, a_2, \dots a_n$ for some $n$ such that $\sum_i a_i = 1$,

$$
f(a_1x_1 + a_2x_2 + \dots a_nx_n) \leq a_1f(x_1) + a_2f(x_2) + \dots + a_nf(x_n)
$$

To show the above result, you can use induction, by using the definition of a convex function as the base case. Assuming the $k^{\text{th}}$ case is true, let's consider the $(k + 1)^{\text{th}}$ case. 

By definition we have $a_1 + a_2 + \dots + a_k + a_{k + 1} = 1$. Defining $b := a_1 + a_2 + ... + a_k$, we have $b + a_{k+1} = 1$.

We now look at the right hand side:

$$
a_1f(x_1) + a_2f(x_2) + \dots + a_kfx_k) + a_{k+1}f(x_{k+1})) = a_{k+1}f(x_{k+1})) + b\sum_i^{k}\frac{a_i}{b}f(x_i)
$$

The last term $\sum_i^{k}\frac{a_i}{b}f(x_i)$, if we cast $\frac{a_i}{b} = a'_i$, you get $\sum_i^k \frac{a_i}{b} = 1$. So basically, the extreme right term can be bounded by the $n=k$ case which we are assuming to be true as part of our induction proof. 

So now, you have 

$$
\begin{aligned}
&a_1f(x_1) + a_2f(x_2) + \dots + a_kfx_k + a_{k+1}f(x_{k+1}) \\
&= a_{k+1}f(x_{k+1}) + b\sum_i^{k}\frac{a_i}{b}f(x_i) \\
&\geq a_{k+1}f(x_{k+1}) + bf(\sum_i^ka'_ix_i) \leftarrow \text{And this is basically the 2 point case} \\
&\geq f(a_{k+1}x_{k+1} + b\sum_i^{k}a'_ix_i) \\
&= f(\sum_i^{k+1}a_ix_i) \\
\end{aligned}
$$

Thus proving the $k+1$ case, and therefore the statement. Look at the [Elements of Information Theory](https://www.amazon.co.uk/Elements-Information-Theory-Telecommunications-Processing/dp/0471241954) if you don't trust what I've written here.


#### Putting it all together
You notice that the definition of probability basically matches that of $a_i$ in the previous part. Substituting $a$ for $p$, you get 

$$
f(p_1x_1 + p_2x_2 + \dots p_nx_n) = f(\mathbb{E}[x]) \leq p_1f(x_1) + p_2f(x_2) + \dots + p_nf(x_n)
$$

$$
p_1f(x_1) + p_2f(x_2) + \dots + p_nf(x_n) = \mathbb{E}[f(x)]
$$

and that's where you get Jensen's inequality from 

$$
\begin{align}
&f(p_1x_1 + p_2x_2 + \dots p_nx_n) \leq p_1f(x_1) + p_2f(x_2) + \dots + p_nf(x_n) \\
&\implies f(\mathbb{E}[x]) \leq \mathbb{E}[f(x)]
\end{align}
$$

This will hold for the continuous case as well.

For _concave_ functions $f$, it is the opposite:

$$
f(\mathbb{E}[x]) \geq \mathbb{E}[f(x)]
$$

In particular, let's remember that logarithmic functions are concave, so 

$$
\text{log}(\mathbb{E}[x]) \geq \mathbb{E}[\log x]
$$

### Mutual information, entropy et al

To understand the equations, we need some more background on mutual information, entropy etc. 

Again, refer back to the [Elements of Information Theory](https://www.amazon.co.uk/Elements-Information-Theory-Telecommunications-Processing/dp/0471241954) book for a more thorough treatment. 

#### Definitions and some results

The entropy H(X) of a random variable X is defined by 

$$
H(X) = -\sum_{x\in \mathcal{X}}p(x)\log p(x)
$$

which is also sometimes written as $H(p)$ because we're all friends here. 

We can also extend this definition to a pair of random variables. There's nothing really "new" because you can consider $(X, Y)$ to be a single vector-valued random function, but it's still useful to write it all out:

>Definition: The joint entropy $H(X, Y)$ of a pair of discrete random. variables (X, Y) with a joint distribution $p(x,y)$ is
>
>$$
 H(X, Y) = - \sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}}p(x, y) \log p(x, y)
$$

Another thing to remember is the definition of conditional entropy

> Definition: If $(X, Y) \sim p(x, y), then the conditional entropy $H(Y|X)$ is defined as
> 
> $$
 H(Y | X) = \sum_{x \in \mathcal{X}} p(x)H(Y|X=x)
$$

> $$
 = - \sum_{x \in \mathcal{X}}p(x) \sum_{y\in \mathcal{Y}}p(y|x)\log p(y|x)
 $$

> $$
= - \sum_{x \in \mathcal{X}}\sum_{y\in \mathcal{Y}}p(x, y)\log p(y|x)
 $$

> $$
= \mathbb{E} \log p(Y|X)
$$

One last thing to also remember is the chain rule of entropy, which is as follows: 

$$
H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
$$

Proof:

$$
\begin{aligned}
H(X,Y) &= -\sum\sum p(x,y)\log p(x,y) \\
&= -\sum\sum p(x,y)\log p(x)p(y|x) \\
&= -\sum\sum p(x,y)\log p(x) - \sum\sum p(x,y)\log p(y|x) \\
&= -\sum p(x)\log p(x) - \sum\sum p(x,y)\log p(y|x) \\
&= H(X) + H(Y|X)
\end{aligned}
$$

The corollary of this is 

$$
H(X, Y | Z) = H(X | Z) + H(Y | X, Z)
$$

Ok, so that is a big chunk. 

Let's move along a bit quicker now. The Kullback-Leibler distance between 2 distributions $p(x), q(x)$ is given by: 

$$
\begin{aligned}
D(p||q) &= \sum_{x\in\mathcal{X}} p(x)\log\frac{p(x)}{q(x)} \\
&= E_p\log\frac{p(X)}{q(X)}
\end{aligned}
$$

With this definition, we can move on to what mutual information is. 

>The mutual information between 2 random variables $X, Y$ with a join probability mass function $p(x, y)$ and marginals $p(x), p(y)$ is the KL distance between the joint distribution and the product distribution $p(x)p(y)$

In equations, this means: 

$$
\begin{aligned}
I(X;Y) = D(p(x, y) || p(x)p(y)) &= \sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}} p(x,y)\log\frac{p(x,y)}{p(x)p(y)} \\
&= D(p(x,y)||p(x)p(y)) \\
&= E_{p(x,y)}\log\frac{p(X,Y)}{p(X)p(Y)}
\end{aligned}
$$

And if you keep turning the crank, this basically becomes 

$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)
$$

There is another definition we need to talk about - the conditional mutual information of random variables $X, Y$ given a third variable $Z$, and this is defined by: 

$$
\begin{aligned}
I(X;Y|Z) &= H(X|Z) - H(X|Y,Z) \\
&= E_{p(x,y,z)}\log\frac{p(X,Y|Z)}{p(X|Z)p(Y|Z)}
\end{aligned}
$$

#### Chain rules

We're nearly there! Let's go through some chain rules now. 

Chain rules for entropy is basically the following:
Let $X_1, X_2, \dots X_n$ be drawn according to $p(x_1, x_2, \dots x_n)$. Then you'll have

$$
H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^n H(X_i|X_{i-1}, \ldots, X_1).
$$

The proof of this is if you write 

$$
\begin{aligned}
p(x_1,\ldots,x_n) &= \prod_{i=1}^n p(x_i|x_{i-1},\ldots,x_1) \\
\implies & H(X_1, X_2,\ldots, X_n) \\
&= -\sum_{x_1,x_2,\ldots,x_n} p(x_1,x_2,\ldots,x_n)\log p(x_1,x_2,\ldots,x_n) \\
&= -\sum_{x_1,x_2,\ldots,x_n} p(x_1,x_2,\ldots,x_n)\log\prod_{i=1}^n p(x_i|x_{i-1},\ldots,x_1) \\
&= -\sum_{x_1,x_2,\ldots,x_n}\sum_{i=1}^n p(x_1,x_2,\ldots,x_n)\log p(x_i|x_{i-1},\ldots,x_1) \\
&= -\sum_{i=1}^n\sum_{x_1,x_2,\ldots,x_n} p(x_1,x_2,\ldots,x_n)\log p(x_i|x_{i-1},\ldots,x_1) \\
&= -\sum_{i=1}^n\sum_{x_1,x_2,\ldots,x_i} p(x_1,x_2,\ldots,x_i)\log p(x_i|x_{i-1},\ldots,x_1) \\
&= \sum_{i=1}^n H(X_i|X_{i-1},\ldots,X_1). \quad \square
\end{aligned}
$$

Similarly, for mutual information you have

$$
I(X_1,X_2,\ldots,X_n;Y) = \sum_{i=1}^n I(X_i;Y|X_{i-1},X_{i-2},\ldots,X_1).
$$

You can prove this using the same cranks, or just refer to the book™. 

## Back to the equation

Ok, that was a lot of equations, but there's going to be more!

Let us remember what we're trying to do again. We want to show this: 

$$
\begin{aligned}
&I(\hat{y}^*;\mathcal{D}|z^*,\theta) \\
&\geq I(x^*;\hat{y}^*|\theta,z^*) \\
&= I(x^*;\hat{y}^*|\theta) - I(x^*;z^*|\theta) + I(x^*;z^*|\hat{y}^*,\theta) \\
&\geq I(x^*;\hat{y}^*|\theta) - I(x^*;z^*|\theta) \\
&= I(x^*;\hat{y}^*|\theta) - \mathbb{E}_{p(x^*)q(z^*|x^*,\theta)}\left[\log\frac{q(z^*|x^*,\theta)}{q(z^*|\theta)}\right] \\
&\geq I(x^*;\hat{y}^*|\theta) - \mathbb{E}\left[\log\frac{q(z^*|x^*,\theta)}{r(z^*)}\right] = I(x^*;\hat{y}^*|\theta) - \mathbb{E}[D_{KL}(q(z^*|x^*,\theta)||r(z^*))]
\end{aligned}
$$

for the following plate diagram:

![[Pasted image 20250207142509.png]]

The first non-obvious part is showing the first jump itself: 

$$
I(\hat{y}^*;\mathcal{D}|z^*,\theta) \geq I(x^*;\hat{y}^*|\theta,z^*)
$$

How does this come about?

We first have to remember the following: 

$$
\begin{aligned}
I(\y;\D|\z,\theta) &= H(\y |\theta, \z) - H(y|\D, \theta, \z) \\
I(\y;\x|\D,\theta, \z) &= H(\y | \D, \theta, \z) - H(y| \x, \D, \theta, \z) \\
I(\y;\x, \D |\theta, \z) &= H(\y |\theta, \z) - H(y| \x, \D, \theta, \z) \\
\implies I(\y;\x, \D |\theta, \z) &= I(\y;\x|\D,\theta, \z) + I(\y;\D|\z,\theta)
\end{aligned}
$$

But if you look at the plate diagram, you have 

$$
I(\y;\x|\D,\theta, \z) = 0
$$

this means that we can now write

$$
\begin{aligned}
I(\hat{y}^*;\mathcal{D}|z^*,\theta) &= I(\hat{y}^*;\mathcal{D}|z^*,\theta) + I(\hat{y}^*;x^*|\mathcal{D},\theta,z^*) \\
&= I(\hat{y}^*;x^*,\mathcal{D}|\theta,z^*) \leftarrow \text{Now what?}\\
\end{aligned}
$$

To further carry on, we need to prove some intermediate results. The first is that $I(X; Y|Z) = I(Y; X|Z)$. This is because: 

$$
\begin{align}
I(X; Y|Z) &= H(X|Z) - H(X|Y,Z) \\
&= H(X,Z) - H(Z) - (H(X, Y, Z) - H(Y, Z)) \\
&= H(Y,Z) - H(Z) - ((H(X, Y, Z) - H(X, Z))) \\
&= H(Y|Z) - H(Y|X, Z) \\
&= I(Y; X|Z)
\end{align}
$$

The next one to show is the following:

$$
I(X; Y,Z|W) = I(X; Y|W) + I(X; Z|Y, W)
$$

Why does this work?
Expanding out the right hand side:

$$
\begin{align}
I(X; Y,Z |W) &= H(X|W) - H(X|Y,Z,W) \\
&= H(X|W) - H(X|Y, W) + H(X|Y, W) - H(X|Y,Z,W) \\
&= I(X; Y|W) + I(X; Z|Y, W)
\end{align}
$$

which is the left hand side. 

Let's go back to where we left off - 

$$
\begin{aligned}
&I(\hat{y}^*;x^*,\mathcal{D}|\theta,z^*) \leftarrow \text{Use second result to expand}\\
&= I(\y;\x|\theta, \z) + I(\y; \D|\x, \theta, \z) \leftarrow \text{Use first result to swap around variables in first term}\\
&= I(x^*;\hat{y}^*|\theta,z^*) + I(\hat{y}^*;\mathcal{D}|x^*,\theta,z^*)
\end{aligned}
$$


Giving a final:

$$
\begin{aligned}
I(\hat{y}^*;\mathcal{D}|z^*,\theta) &= I(\hat{y}^*;\mathcal{D}|z^*,\theta) + I(\hat{y}^*;x^*|\mathcal{D},\theta,z^*) \\
&= I(\hat{y}^*;x^*,\mathcal{D}|\theta,z^*) \\
&= I(x^*;\hat{y}^*|\theta,z^*) + I(\hat{y}^*;\mathcal{D}|x^*,\theta,z^*) \\
&\geq I(x^*;\hat{y}^*|\theta,z^*)
\end{aligned}
$$

and thus, now we can understand Equation 2 from the paper in its entirety: 

$$
\begin{aligned}
&I(\hat{y}^*;\mathcal{D}|z^*,\theta) \\
&\geq I(x^*;\hat{y}^*|\theta,z^*) \\
&= I(x^*;\hat{y}^*|\theta) - I(x^*;z^*|\theta) + I(x^*;z^*|\hat{y}^*,\theta) \\
&\geq I(x^*;\hat{y}^*|\theta) - I(x^*;z^*|\theta) \\
&= I(x^*;\hat{y}^*|\theta) - \mathbb{E}_{p(x^*)q(z^*|x^*,\theta)}\left[\log\frac{q(z^*|x^*,\theta)}{q(z^*|\theta)}\right] \\
&\geq I(x^*;\hat{y}^*|\theta) - \mathbb{E}\left[\log\frac{q(z^*|x^*,\theta)}{r(z^*)}\right] = I(x^*;\hat{y}^*|\theta) - \mathbb{E}[D_{KL}(q(z^*|x^*,\theta)||r(z^*))]
\end{aligned}
$$

where $r(\z)$ is a variational approximation of the marginal $q(\z|\theta)$ 

What does this all mean?

The whole aim of meta-regularisers It means that if you want to force the model to learn from the training dataset for each $\T_i$, one "tractable" way you can do it is by increasing the mutual information between $\x$ and $\y$ while at the same time restricting the information flow from $\x$ to $\y$ through a stochastic variable $z$.

Equation 2 from the paper is just one of many equations presented by the authors of the paper, but now, equipped with the various relationships between entropy, mutual information etc., you too can understand the authors' motivation of setting up the optimization functions and training losses they way they have set it up!
