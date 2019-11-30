from __future__ import division
from collections import Counter
from codecs import open
import math


def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


def train_nb(documents, labels):
    # Create three counters, used to keep track of each word we come across, and how many times we do
    vocabulary = Counter()
    positive = Counter()
    negative = Counter()

    # Keep track of how many total lines there are, and how many there are for +ve and -ve
    # (used later to determine probability of +ve or -ve)
    total_count = 0
    positive_count = 0
    negative_count = 0

    # for each line, increment the above count variables. then, for each word
    # in the line, put it in the vocabulary counter, and in the +ve or -ve counter (depending on its label)
    for doc, label in zip(documents, labels):
        total_count += 1
        if label == "pos":
            positive_count += 1
        if label == "neg":
            negative_count += 1
        for w in doc:
            vocabulary[w] += 1
            if label == "pos":
                positive[w] += 1
            if label == "neg":
                negative[w] += 1

    # Add one of each word in the vocabulary to each the positive and negative counters, (add-1 smoothing)
    for w in vocabulary:
        positive[w] += 1
        negative[w] += 1

    # represents the overall probability of falling on a positive or negative review
    # (using the aforementioned count variables)
    positive_probability = positive_count / total_count
    negative_probability = negative_count / total_count

    # To find the probability of each word, lets first find the total number of words for each
    positive_total_words = 0
    negative_total_words = 0

    for w in positive:
        positive_total_words += positive[w]

    for w in negative:
        negative_total_words += negative[w]

    # Then, we'll create new dictionaries, mapping each word to its probability
    positive_word_probabilities = {}
    negative_word_probabilities = {}

    for w in positive:
        positive_word_probabilities[w] = positive[w] / positive_total_words

    for w in negative:
        negative_word_probabilities[w] = negative[w] / negative_total_words

    # return a tuple with everything needed to test words out!
    return positive_word_probabilities, negative_word_probabilities, positive_probability, negative_probability


def score_doc_label(document, label, library):
    # Extract the elements necessary to classify, from the library tuple
    positive_word_probabilities = library[0]
    negative_word_probabilities = library[1]
    positive_probability = library[2]
    negative_probability = library[3]

    score = 0
    if label == "pos":
        # Add the log of the overall probability of the doc being +ve
        score += math.log(positive_probability)
        # Add the log of the probability of the word given a +ve doc to the score
        for w in document:
            if w in positive_word_probabilities:
                score += math.log(positive_word_probabilities[w])
        return score
    if label == "neg":
        # Add the log of the overall probability of the doc being -ve
        score += math.log(negative_probability)
        # Add the log of the probability of the word given a -ve doc to the score
        for w in document:
            if w in negative_word_probabilities:
                score += math.log(negative_word_probabilities[w])
        return score


def classify_nb(document, library):
    score_pos = score_doc_label(document, "pos", library)
    score_neg = score_doc_label(document, "neg", library)

    # Compare both scores and return the highest one
    if score_pos > score_neg:
        return "pos"
    if score_neg > score_pos:
        return "neg"


def classify_documents(docs, library):
    predicted_labels = []
    for d in docs:
        predicted_labels.append(classify_nb(d, library))
    return predicted_labels


def accuracy(true_labels, guessed_labels, eval_docs):
    correct = 0
    pos_correct = 0
    neg_correct = 0

    total_pos = 0
    total_neg = 0
    for true, guessed, doc in zip(true_labels, guessed_labels, eval_docs):
        if true == "pos":
            total_pos += 1
        else:
            total_neg += 1

        if true == guessed:
            correct += 1
            if true == "pos":
                pos_correct += 1
            else:
                neg_correct += 1
        else:
            print(true, guessed, doc, sep=", ")
    return correct / len(true_labels), (pos_correct / total_pos), (neg_correct / total_neg)


# ---------------- start of main ---------------------- #
all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

trained_library = train_nb(train_docs, train_labels)

predicted = classify_documents(eval_docs, trained_library)

res = accuracy(eval_labels, predicted, eval_docs)
print("Overall accuracy:", res[0])
print("Positive accuracy:", res[1])
print("Negative accuracy:", res[2])
