from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import string
import sys

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# Creating a pandas dataframe from the dictionary data
def create_df(file):
    data_list = []
    # Adding objects and their labels to list, then adding those to a dataframe: each row is one review and contains a column for the label and column for the text
    for label, object in zip(file['labels'], file['objects']):
        data_list.append({'labels': label, 'objects': object})
    df = pd.DataFrame(data_list)
    # Removing all punctuation and making all words lowercase
    df.objects = df.objects.apply(lambda t: t.lower().translate(str.maketrans('', '', string.punctuation)))
    return df

# Percentage of all reviews that are deceptive: prior probability
def decep_perc(df):
    # Count the number of reviews where the labels column is deceptive
    decep_count = df['labels'].value_counts().get('deceptive', 0)
    # Count the number of total reviews
    total_objects = len(df)
    decep_perc = (decep_count / total_objects)
    return decep_perc

# Return a dataframe containing every common word between truthful and deceptive reviews as columns, and deceptive and truthful as the 2 rows.
# It finds the probability of each common word appearing in the deceptive words, and appearing in the truthful words.  This finds the probability of each word given truthful and p of each word given deceptive.
def bag_of_words(df):
    # scikit-learn CountVectorizer efficiently parses through words and finds word counts
    vectorizer = CountVectorizer()
    # This will find the word counts of each unique word for each review in the input training df by creating a matrix, and it also extracts the column names for use later.
    # It then converts the matrix to a pandas df.
    transformed = vectorizer.fit_transform(df['objects'])
    vectorized_df = pd.DataFrame(transformed.toarray(), columns=vectorizer.get_feature_names_out())
    # This will group all the word counts for each word (column) by deceptive and truthful
    grouped = vectorized_df.groupby(df['labels']).sum()
    # This will remove all the words that are not common to both deceptive and truthful reviews.  
    grouped = grouped.loc[:, (grouped != 0).all()]
    # For each word (column), this will calculate the ratio of deceptive words that are this word, and the ratio of truthful words that are this word.
    word_perc = grouped.div(grouped.sum(axis=1), axis=0)
    return word_perc

# classifier : Train and apply a bayes net classifier
#
# This function takes a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# returns a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated class label for test_data["objects"][i]


def classifier(train_data, test_data):
    # Create a dataframe from the training data and the bag of words which then finds probabilities of each word in the training data
    train = create_df(train_data)
    training_data = bag_of_words(train)
    # Grab the list of unique words from the training data
    training_words = set(training_data.columns)
    # Initialize a list which will contain all the class assignments corresponding to each word from the test data, this will be returned
    test_data_labels = []
    # For each review in the test data
    for review in test_data['objects']:
        # Turn all strings to lowercase and remove punctuation, and split all the words in the review into a set of strings and then only keep the words from the test review that are present in the training data
        review_set = review.lower().translate(str.maketrans('', '', string.punctuation))
        review_set = review_set.split()
        common = list(set(review_set).intersection(training_words))
        # Using the training data, find the probability of finding each common word in the deceptive words and the truthful words
        deceptive_prob = [training_data.iloc[0][i] for i in common]
        truthful_prob = [training_data.iloc[1][i] for i in common]
        # Using logs because the number computations can get extremely small and decrease efficiency.
        # With logs, use Bayes' Thereom to find log(P(word 1|deceptive)) + log(P(word 2|deceptive)) + ... + log(P(deceptive)).  Do the same for truthful.
        decep_calculation = sum([np.log(p) for p in deceptive_prob]) + np.log(decep_perc(train))
        truth_calculation = sum([np.log(p) for p in truthful_prob]) + np.log(1-decep_perc(train))
        # If value for deceptive is greater than truthful, then append "deceptive" to the test_data_labels list, which will correspond that label for that review.
        # Do the same for truthful reviews, then continue looping over all the reviews in the test data, then return the test_data_labels list.
        if decep_calculation >= truth_calculation:
            test_data_labels.append("deceptive")
        else: 
            test_data_labels.append("truthful")

    return test_data_labels


train_data = load_file("deceptive.train.txt")
test_data = load_file("deceptive.test.txt")
if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
    raise Exception("Number of classes should be 2, and must be the same in test and training data")

# make a copy of the test data without the correct labels, so we know the classifier isn't cheating
test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

results = classifier(train_data, test_data_sanitized)

# calculate accuracy
correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
