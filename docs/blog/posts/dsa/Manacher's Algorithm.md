---
draft: true
date: 2023-01-31
slug: manacher-algo
categories:
  - algorithms
tags:
  - algorithms
---
Super annoying algorithm, but it has uses in [bioinfomatics](https://stackoverflow.com/questions/23861436/real-life-situations-using-palindrome-algorithm). 

Here's the task:
Given a string `s`, return the longest palindromic substring  in `s`.

Example inputs and outputs:

**Input:** s = "babad"
**Output:** "bab"
**Explanation:** "aba" is also a valid answer.

**Input:** s = "cbbd"
**Output:** "bb"

To build some intuition for how this algorithm works, let's see an example of a brute force implementation. In this implementation, given a string `_str`, we want to see for any `i`, what is the maximum length of a palindrome centred around the character `_str[i]`. Something simple would be something like this: 

```python
def manacher_brute_force(_str: str) -> list:
  _str_len = len(_str)
  out = [0 for _ in range(_str_len + 2)]
  # make sure first and last chars are different and do not
  # happen in _str itself
  _str = f"${_str}^"

  for i in range(1, _str_len + 1):

    while _str[i - out[i]] == _str[i + out[i]]:
      out[i] += 1
  return out

if __name__ == "__main__":
  print(manacher_brute_force("abbcbba"))
```

the output will be something like `[0, 1, 1, 1, 4, 1, 1, 1, 0]`

But in this approach, when we sweep from left to right, we are not using the work we have already done to the left of `i`. This is where Manacher comes in. 

## What don't we have to search?
We can exploit the nature of a palindrome - assume we have a palindrome of length `l` centred around index `i`, and say we take 2 indexes `i'` and `i''` that are distance `d` left and right to `i` respectively s.t. `d < l`, then we basically know that any palindrome that is centred around `i'` will also likely be centred around `i''`!

From this simple observation, we can already amend the inner `while` loop so that we are not searching *all* of the characters to the left and right of a particular index `i''` of an index that is to the right of an index `i` we have already done work for.  

The rest of the complexity comes from the need to handle the case where the borders of the inner palindrome reaches the border of the outer palindrome. All you have to do there is make sure you always check whenever you go beyond the borders of the longest current palindrome. 

```python

def manacher_odd(_str: str) -> list:
	_str_len = len(_str)
	out = [0 for _ in range(_str_len + 2)]
	# make sure first and last chars are different and do not
	# happen in _str itself
	_str = f"${_str}^"
	l, r = 1, 1
	for i in range(1, _str_len + 1):
		dist_to_border = r - i
		inner_palindrome_len = min(dist_to_border, out[l + dist_to_border])
		out[i] = max(0, inner_palindrome_len)
		while _str[i - out[i]] == _str[i + out[i]]:
		  out[i] += 1
		if i + out[i] > r:
			l = i - out[i]
			r = i + out[i]
	return out
```

Code is a translation of the cpp code [here](https://cp-algorithms.com/string/manacher.html)

NOTE:

The above algorithm is only for odd length. In practice, you can make any string odd length by doing something like 

```python
_str = "£".join(_str)
_str = f"#{_str}^"
```

the `.join` adds `n-1` characters to the string, so the total number of characters will be odd, since `even + odd = odd`
## Where does the time saving come from that makes it linear?
In the brute force method, consider the number of times a character at index `i` is compared to some other character. You will quickly realise it is `O(n)`, and that is where the overall $O(n^2)$ complexity for the brute force method comes from. 

But with Manacher's algorithm, the while loop is no longer independent of the outer for loop. The outer loop is keeping track of the `centre` of palindromes and is always increasing. We only do additional comparison operations when `r` variable (i.e. the rightmost boundary, the `centre + radius` value) increases - and this quantity never decreases in value! Therefore, the total number of operations in the outer and in the inner loop adds to `n`. 

## Final:

```python

def manacher(s: str) -> str:
        if len(s) <= 1:
            return s

        s = f"#{'#'.join(s)}#"
        s_len = len(s)
        out = [0 for _ in range(s_len)]
        max_radius = 1
        max_str = s[1]

        l, r = 1, 1
        for i in range(1, s_len - 1):
            dist_to_edge = r - i
            allowable_dist = min(dist_to_edge, out[l + dist_to_edge])
            out[i] = max(0, allowable_dist)
            while i - out[i]>= 0 and i + out[i] < s_len and s[i - out[i]] == s[i + out[i]]:
                out[i] += 1
            if i + out[i] > r:
                r = i + out[i]
                l = i - out[i]
            if out[i] > max_radius:
                max_radius = out[i]
                max_str = s[l + 1 : r].replace("#", "") # if you want s[l: r + 1] you need to offset in the while loop like so: 
                # while i - out[i] - 1>= 0 and i + out[i] + 1 < s_len and s[i - out[i] - 1] == s[i + out[i] + 1]
        return max_str
```

or if you want to use the original `manacher_odd` kind of notation:

```python
	def manacher(s: str) -> str:
		if len(s) <= 1:
            return s

        s = f"£#{'#'.join(s)}#^"
        s_len = len(s)
        out = [0 for _ in range(s_len)]
        max_radius = 1
        max_str = s[1]

        l, r = 1, 1
        for i in range(1, s_len - 1):
            dist_to_edge = r - i
            allowable_dist = min(dist_to_edge, out[l + dist_to_edge])
            out[i] = max(0, allowable_dist)
            while s[i - out[i]] == s[i + out[i]]:
                out[i] += 1
            if i + out[i] > r:
                r = i + out[i]
                l = i - out[i]
            if out[i] > max_radius:
                max_radius = out[i]
                max_str = s[l + 1 : r].replace("#", "")
        return max_str
```
