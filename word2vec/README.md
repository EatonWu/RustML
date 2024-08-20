# Word2Vec

This isn't part of Charniak, but rather Stanford's NLP class.
Word2Vec is a neural network based model for learning word embeddings.

## Constructing word vectors

Word vectors are used downstream in NLP for many, many things.
A word vector represents a words in a high-dimensional space,
where each dimension represents a feature/meaning of a word.

There are two primary models used for Word2Vec NLP processing:

## Skip-Gram

Using some center word, predict the context words.

For example, given the sentence:

```
The quick brown fox jumps over the lazy dog.
```

The dataset would maybe look like:

(quick, the),
(quick, brown),
(brown, quick),
(brown, fox),
...
## Continuous Bag of Words (CBOW)

Using some outer words, predict the center word.

For example,
([the, brown], quick),
([quick, fox], brown),
...

## Bigrams vs Unigrams

We can represent valid sentences (as opposed to invalid sentences) as probabilities.

In a unigram model, the probability of a valid sentence is equal to the
product of the probabilities of each word in the sentence.

This is obviously not going to be very good, so instead, we can use a bigram model,
in which the probability of a sentence being valid is equal to the product of
the probabilities of each word given the previous word P(w_i | w_{i-1}).