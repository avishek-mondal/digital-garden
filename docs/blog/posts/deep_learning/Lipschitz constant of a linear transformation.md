---
title: Lipschitz constant of a linear transformation
draft: false
date: 2024-08-05
---

DISCLAIMER: If you spot an error, please feel free to email me. 


The "Understanding Deep Learning" book has recently come out (you can look at it [here](https://udlbook.github.io/udlbook/), this post refers to the 2023-05-08 version), and is a great resource. In the appendices, it contains several statements which can be non-obvious to those of us who have been out of touch with linear algebra in our day jobs. 

Here is one such statement: 

"The Lipschitz constant of a linear transformation $f[z] = Az + b$ is equal to the maximum eigenvalue of the matrix A."



This is not an obvious result at all. Let us try and break this down step by step.

<!-- more -->
## Definitions

### Lipschitz constant
A function $f[z]$ is Lipschitz continuous if for all $z1,z2$:

$$
|f[z1] − f[z2]| ≤ β|z1 − z2|,
$$

The value $\beta$ is called the Lipschitz constant. Both definitions are taken from the UDL book in the link above.

### Vector norm
The following definitions are lifted from this very useful resource [here](https://www.math.drexel.edu/~foucart/TeachingFiles/F12/M504Lect6.pdf) (Definition 3)



Let $\mathcal{V}$ be a vector space over a field $\mathbb{R}$. A function $\lvert{.}\rvert : \mathcal{V} → \mathbb{R}$ is called a (vector) norm if


1. $\lvert x \rvert ≥ 0$ for all $x ∈ \mathcal{V}$ , with equality iff x = 0, [positivity]
2. $\lvert λx \rvert = \lvertλ \rvert \lvert x \rvert$ for all λ ∈ $\mathbb{K}$ and x ∈ V , [homogeneity] (reminder $\mathbb{K} ∈ \{\mathbb{R}, \mathbb{C} \}$, i.e. either real or complex)
3. $\lvert x + y \rvert ≤ \lvert x \rvert + \lvert y \rvert$ for all x, y ∈ V . [triangle inequality]

An example of such a norm would be an expression as follows for $p \geq 1$: 

$$
|x|_p = [|x_1|^p + |x_2|^p + |x_3|^p + ... ]^{\frac{1}{p}}
$$


### Matrix norm
Also from the above resource (Definition 7)

A function 
$|| .||: \mathcal{M} \rightarrow \mathbb{C}$ 
is a matrix norm if:

1. 
$||A|| \geq 0$ for all 
$A \in \mathcal{M}$with equality iff x = 0 [positivity]

2. 
$||\lambda A|| = |\lambda| ||A||$ for all
$\lambda \in \mathbb{C}$and
$A \in \mathcal{M}$[homogeneity]

3.$||A + B|| \leq ||A|| + ||B||$ for all
$A, B \in \mathcal{M}$ [triangle inequality]

4.$||AB|| \leq ||A|| ||B||$ for all
$A, B \in \mathcal{M}$ [submultiplicativity]

The last property, submultiplicativity, deserves some more attention. It uses a more general definition of matrix norm which is 

$$
||A|| = \text{sup}\{|Ax|: x \in \mathcal{V}, |x| \leq 1 \} \\ 
= \text{sup}\{\frac{|Ax|}{|x|}: x \in \mathcal{V}, x \neq 0 \}
$$

(lots of resources use this definition, for clarification of max vs sup, look [here](https://math.stackexchange.com/questions/3575425/why-supremum-not-max-in-definition-of-matrix-norm)).

An intermediate result we can now prove is the following (or you could also just look [here](https://math.stackexchange.com/questions/1513399/ax%E2%89%A4-a-x-space-forall-x-in-mathbbrn-rudins-principles)):

$$
|Ax| \leq ||A|| |x|, \, \forall x \in \mathbb{R}^n
$$

The proof in the top answer of the above link from Stackoverflow is copied below for ease of reference: 

By definition $‖𝐴‖= \text{sup} \{|Ax|, x ∈ \mathbb{R}^n, |x| ≤ 1 \} $ and hence for any 
$x \in \mathbb{R}^n$ such that 
$|x| \leq 1$ we must have by the definition of supremum that 
$|Ax| \leq ||A||$

The $x = 0$ case is trivial.

For the $x \neq 0$ case: 

Let 
$y = \frac{x}{|x|}$.
It follows that $|y| = 1$, and hence $|Ay| \leq ||A||$ from above.

Now we have 

$$
|Ay| = | A \frac{x}{|x|} | = \frac{|Ax|}{|x|}
$$

Therefore, since $|x|>0$,
we have $|Ax|=|Ay||x|≤ ||A|| |x|$

## Tying it all together

Now, we can look at the Lipschitz constant of a linear transformation $f[z] = Az + b$.

$$
|f[z1] − f[z2]| = |Az_1 - Az_2| \\
= |A(z_1 - z_2)| \\
\leq ||A|| |z_1 - z_2|
$$

The last step is using the same intermediate property we proved when showing the proof for submultiplicativity.

This is the general case - the Lipschitz constant of a linear transform will be the matrix norm of the matrix $A $$.

We need one more step to prove the claim in the book, and I think that involves deviating away from the general case.

I am going to be following some of the material [here](https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec7matrixnorm.pdf). When the 2-norm is used, the induced matrix operator is the following (called the spectral norm):

$$
||A||_2 = \text{max}_{x \neq 0} \{ \frac{|Ax|_2}{|x|_2} \}
$$

It is a known result for the 2-norm that 
$|y|_2^2 = y^Ty$

So let us take the square of the induced matrix operator.

$$
||A||_2^2 = \text{max}_{x \neq 0} \{ \frac{|Ax|_2^2}{|x|_2^2} \} \\
= \text{max}_{x \neq 0} \{ \frac{x^TA^TAx}{x^Tx} \}
$$

We will need to use concepts from singular value decomposition (SVD). [This](https://math.mit.edu/classes/18.095/2016IAP/lec2/SVD_Notes.pdf) resource might help. The "trick" is to write A as

$$
A = U\Sigma V^T
$$

and then write $A^TA = (U\Sigma V^T)^T (U\Sigma V^T) = V \Sigma^T \Sigma V$, where $\Sigma $is a diagonal matrix with the eigenvalues of A in the diagonal.

Let $\lambda_1 $be the largest eigenvalue of A. Then, the maximum value of the ratio of $\frac{x^TA^TAx}{x^Tx}$ can be shown to be $\lambda_1^2$.

That was the final step - showing 
$||A||_2^2 = \lambda_1^2$,
i.e. $||A||_2 = \lambda_1$

So therefore, 

$$
|f[z1] − f[z2]|_2 = |Az_1 - Az_2|_2 \\
= |A(z_1 - z_2)|_2 \\
\leq ||A||_2 |z_1 - z_2|_2 \\
= \lambda_1 |z_1 - z_2|
$$

Other resources: 
1. I found [this](https://www.math.drexel.edu/~foucart/TeachingFiles/F12/M504Lect5.pdf) from Drexel University to also be quite helpful
