
# coding: utf-8

# In[2]:


import ntlk
from ntlk.tokenize import word_tokenize, sent_tokenize
sentence = "Hello this is natural language processing. Module is NTLK"
word_tokenize(sentence)


# In[31]:


documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"We can see the shining sun, the bright sun",
)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(tfidf_matrix.shape)
print(tfidf_matrix)

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

