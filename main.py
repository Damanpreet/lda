import numpy as np
# import lda
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

print(X.shape)
print(X.sum())

model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)

# initializes the model parameters with the initial probabilties.
# runs the model for given no of iterations and get the model parameters.
# access the model resuls using topic_word_ or components_
model.fit(X)  # model.fit_transform(X) is also available

# gives the final topic word distribution. can be used for inference.
topic_word = model.topic_word_  # model.components_ also works

n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))