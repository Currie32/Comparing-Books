
# coding: utf-8

# # Comparing Project Gutenberg's 20 Most Popular Books

# In this analysis we will use Word2Vec and Doc2Vec to compare Project Gutenberg's 20 most popular books. Word2Vec will be used to show how the relationship between characters can be drawn within, and between books. Doc2Vec will be used to determine which books are most similar. The 20 books that we will use are:
# 1. A Dolls House by Henrik Ibsen
# 2. A Tale of Two Cities by Charles Dickens
# 3. Adventures of Huckleberry Finn by Mark Twain
# 4. Alices Adventures in Wonderland by Lewis Carroll
# 5. Dracula by Bram Stoker
# 6. Emma by Jane Austen
# 7. Frankenstein by Mary Shelley
# 8. Great Expectations by Charles Dickens
# 9. Grimms Fairy Tales by The Brothers Grimm
# 10. Metamorphosis by Franz Kafka
# 11. Pride and Prejudice by Jane Austen
# 12. The Adventures of Sherlock Holmes by Arthur Conan Doyle
# 13. The Adventures of Tom Sawyer by Mark Twain
# 14. The Count of Monte Cristo by Alexandre Dumas
# 15. The Importance of Being Earnest by Oscar Wilde
# 16. The Picture of Dorian Gray by Oscar Wilde
# 17. The Prince by Nicolo Machiavelli
# 18. The Romance of Lust by Anonymous
# 19. The Yellow Wallpaper by Charlotte Perkins Gilman
# 20. Ulysses by James Joyce
# 
# Further information about Project Gutenberg can be found at http://www.gutenberg.org

# In[386]:

import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gensim
get_ipython().magic('pylab inline')


# In[385]:

# Create a list of all of our book files.
book_filenames = sorted(glob.glob("*.rtf"))
print("Found books:")
book_filenames


# In[4]:

# Read and add the text of each book to corpus_raw.
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# In[6]:

# Tokenize each sentence
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)


# In[406]:

def sentence_to_wordlist(raw):
    '''Remove all characters except letters'''
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[8]:

# Clean the raw_sentences and add them to sentences.
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[395]:

# Take a look at a sentence before and after it is cleaned.
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


# In[10]:

# Find the total number of tokens in sentences
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


# # Word2Vec

# In[397]:

# Set the parameteres for Word2Vec
num_features = 300
min_word_count = 20
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-4
seed = 2


# Note: Using a lower min_word_count could make our work below more accurate, however, TSNE crashes my jupyter notebook when I increase the size of all_word_vectors_matrix (below).

# In[398]:

books2vec = w2v.Word2Vec(
    sg=1, #skip-gram
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[399]:

# Build the vocabulary
books2vec.build_vocab(sentences)
print("books2vec vocabulary length:", len(books2vec.vocab))


# In[407]:

books2vec.train(sentences)


# In[202]:

# Create a vector matrix of all the words
all_word_vectors_matrix = books2vec.syn0


# In[201]:

# Use TSNE to reduce all_word_vectors_matrix to 2 dimensions. 
tsne = sklearn.manifold.TSNE(n_components = 2, 
                             early_exaggeration = 6,
                             learning_rate = 500,
                             n_iter = 2000,
                             verbose = True,
                             random_state = 2)


# In[205]:

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# In[206]:

# Create a dataframe to record each word and its coordinates.
points = pd.DataFrame(
    [(word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[thrones2vec.vocab[word].index])
            for word in thrones2vec.vocab
        ]],
    columns=["word", "x", "y"])


# In[207]:

# Preview the points
points.head()


# In[209]:

# Display the layout of all of the points.
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(10, 6))


# In[210]:

def plot_region(x_bounds, y_bounds):
    '''Plot a limited region with points annotated by the word they represent.'''
    slice = points[(x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) & 
                   (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 6))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# In[223]:

# Find the coordinates for Alice - Alice's Adventures in Wonderland
points[points.word == 'Alice']


# In[232]:

plot_region(x_bounds=(-3.3, -2.7), y_bounds=(0.2, 0.4))


# We can see that a number of characters from Alice's Adventures in Wonderland are grouped together.

# In[235]:

# Find the coordinates for (Tom) Sawyer - The Adventures of Tom Sawyer
points[points.word == 'Sawyer']


# In[237]:

plot_region(x_bounds=(-4.5, -3.5), y_bounds=(0.5, 1))


# Similarly, both characters and their first and last names are grouped together: Becky Thatcher, Judge Thatcher, Huck/Hucklebrry Finn, Tom Sawyer, Sid Sawyer, and Aunt Polly.

# Now let's find the most similar words for each given word.

# In[408]:

books2vec.most_similar("monster") 


# In[240]:

books2vec.most_similar("Sherlock")


# In[241]:

books2vec.most_similar("dog")


# In[246]:

books2vec.most_similar("frightened")


# For the most part, the similar words seem to match the given one.

# In[272]:

def nearest_similarity_cosmul(start1, end1, start2):
    '''Find the word that completes the relationship.'''
    similarities = books2vec.most_similar_cosmul(
        positive=[start1, start2],
        negative=[end1])
    end2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return end2


# In[315]:

nearest_similarity_cosmul("Sherlock", "Holmes", "Huck")
nearest_similarity_cosmul("Pip", "Havisham", "Alice")
nearest_similarity_cosmul("good", "evil", "happy")


# Again, we can see expected relationships, such as: 
# - first/last names: Sherlock Holmes - Huck Finn
# - Character Relationships: Pip Pirrip + Miss Havisham - Alice + Caterpillar
# 
# As I was experimenting with different words, I found the third relationship. It's surprised me and I find it rather interesting, so I want to share it with you all.

# # Doc2Vec

# In[339]:

# Read and tag each book into book_corpus
book_corpus = []
for book_filename in book_filenames:
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        book_corpus.append(
            gensim.models.doc2vec.TaggedDocument(
                gensim.utils.simple_preprocess( # Clean the text with simple_preprocess
                    book_file.read()),
                    ["{}".format(book_filename)])) # Tag each book with its filename


# In[410]:

# We can expand the vocabulary by setting the min_count to 3.
# Larger values for iter should improve the model's accuracy.
model = gensim.models.doc2vec.Doc2Vec(size = 300, 
                                      min_count = 3, 
                                      iter = 100)


# In[411]:

model.build_vocab(book_corpus)
print("model's vocabulary length:", len(model.vocab))


# In[421]:

model.train(book_corpus)


# Below we will find the most similar books compared to the given one.

# In[422]:

model.docvecs.most_similar("The_Adventures_of_Tom_Sawyer_by_Mark_Twain.rtf")


# In[423]:

model.docvecs.most_similar("The_Adventures_of_Sherlock_Holmes_by_Arthur_Conan_Doyle.rtf")


# In[424]:

model.docvecs.most_similar("The_Prince_by_Nicolo_Machiavelli.rtf")


# In[425]:

# Find the most similar book for each book
for book in book_filenames:
    most_similar = model.docvecs.most_similar(book)[0][0]
    print("{} - {}".format(book, most_similar))


# I feel quite good about the results that we have here. Books by the same author are often the most similar: Adventures of Huckleberry Finn - The Adventures of Tom Sawyer; Emma - Pride and Prejudice, however The Importance of Being Earnest - The Picture of Dorian Gray did not match up both ways.
# 
# The Prince is the most similar book for four others. An interesting idea for another project would be to try and find what makes this book able to match other books so well.

# # Summary

# I hope that you will agree with me and say that this project is a success. By using Word2Vec we were able to draw comparisons within and between books, and by using Doc2Vec we could compare the overall similarity of books. Some of the examples that I used were 'cherry-picked.' If you try out this project and `nearest_similarity_cosmul()` doesn't always work exactly as you expected, that happened to me as well. Nonetheless, the fact that it does work and with multiple examples is very positive.

# In[ ]:



