import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords # import the stop words list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
TRAINING_DATA_PATH = 'alldata/labeledTrainData.tsv'
TESTING_DATA_PATH = 'alldata/testData.tsv'

vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)


def read_training_data(path):
    raw_train_data = pd.read_csv(path, header=0, delimiter='\t', quoting=3)
    return raw_train_data


def read_testing_data(path):
    raw_test_data = pd.read_csv(path, header=0, delimiter='\t', quoting=3)
    return raw_test_data


def text_processing(raw_text):
    # 1. Remove html
    no_html_text = BeautifulSoup(raw_text, "lxml").get_text()

    # 2. Remove number and punctuation， only letters
    letters_only_text = re.sub('[^a-zA-Z]', ' ', no_html_text)

    # 3. Convert upper case to lower case
    letters_lower_case = letters_only_text.lower().split()

    # 4. Stop words removal
    # stopwords: list to set
    stop_words = set(stopwords.words("english"))
    text_no_stop_words = [w for w in letters_lower_case if not w in stop_words]

    # 5. Convert tokens to string for Bag of Words
    return (" ".join(text_no_stop_words))


def training_data_cleaning_one_line(raw_train_data):

    processed_train_data = text_processing(raw_train_data['review'][0])

    logging.info(processed_train_data)

    return processed_train_data


def training_data_cleaning(raw_train_data):
    logging.info("Cleaning the training data...\n")

    # number of reviews
    training_reviews_num = raw_train_data['review'].size

    processed_train_data = []
    for i in range(0, training_reviews_num):
        # If the index is evenly divisible by 1000, print a message
        if ((i + 1) % 1000 == 0):
            logging.info("Movie Review: No. %d of %d\n" % (i + 1, training_reviews_num))
        processed_train_data.append(text_processing(raw_train_data["review"][i]))
    return processed_train_data


def testing_data_cleaning(raw_test_data):
    logging.info("Cleaning the testing data...\n")

    testing_reviews_num = raw_test_data["review"].size

    processed_test_data = []
    for i in range(0, testing_reviews_num):
        if ((i + 1) % 1000 == 0):
            logging.info("Movie Review: No. %d of %d\n" % (i + 1, testing_reviews_num))
        processed_test_data.append(text_processing(raw_test_data["review"][i]))
    return processed_test_data


def creating_bag_of_words(processed_train_data):
    logging.info("Creating the bag of words...\n")

    training_data_features = vectorizer.fit_transform(processed_train_data)

    # training data features array
    training_data_features = training_data_features.toarray()

    logging.info(training_data_features.shape)

    vocabulary = vectorizer.get_feature_names()

    return training_data_features, vocabulary


def creating_testing_data_features(processed_test_data):
    test_data_features = vectorizer.transform(processed_test_data)

    test_data_features = test_data_features.toarray()

    logging.info(test_data_features.shape)

    return test_data_features


def building_model(training_data_features, test_data_features, raw_train_data, raw_test_data):

    logging.info("Training the random forest model...")

    forest = RandomForestClassifier(n_estimators=100)

    forest = forest.fit(training_data_features, raw_train_data["sentiment"])

    result = forest.predict(test_data_features)

    output = pd.DataFrame(data={"id": raw_test_data["id"], "sentiment": result})

    output.to_csv("result/Bag_of_Words_model.csv", index=False, quoting=3)



training_data = read_training_data(TRAINING_DATA_PATH)

processed_train_data = training_data_cleaning(training_data)

training_data_features, vocabulary = creating_bag_of_words(processed_train_data)

logging.info(training_data_features)

logging.info(vocabulary[:10])

testing_data = read_testing_data(TESTING_DATA_PATH)

processed_test_data = testing_data_cleaning(testing_data)

testing_data_features = creating_testing_data_features(processed_test_data)

building_model(training_data_features, testing_data_features, training_data, testing_data)






