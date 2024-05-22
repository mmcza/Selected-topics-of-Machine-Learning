from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

def main():
    categories = ['sci.space', 'rec.sport.hockey',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories,
                                      shuffle=True,
                                      random_state=42)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print(count_vect.vocabulary_.get('laboratory'))
    print(count_vect.transform(['laboratory']))

    print(count_vect.vocabulary_.get('WZUM'))
    print(count_vect.transform(['WZUM']))

    tfidf_transformer = TfidfTransformer(use_idf=False)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

    docs_new = ['There was a new planet discovered',
                'There was a new organ discovered',
                'OpenGL on the GPU is fast']

    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))

def main2():

    categories = ['sci.space', 'rec.sport.hockey',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories,
                                      shuffle=True,
                                      random_state=42)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(twenty_train.data, twenty_train.target)

    twenty_test = fetch_20newsgroups(subset='test',
                                     categories=categories, shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    print(np.mean(predicted == twenty_test.target))

    cm = confusion_matrix(twenty_test.target, predicted, labels=text_clf.classes_)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=twenty_test.target_names).plot()
    plt.show()


if __name__ == '__main__':
    #main()
    main2()