# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from collections import Counter
import re
import os
import glob
import collections
import pickle
import seaborn as sns
import statistics
import itertools
import timeit
from nltk import agreement
# Parse html entities within text
import html

# expand contractions
import contractions

# ntlk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer # For evaluation

# To convert emojis and emoticons to words
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

# Compare ntlk tokenizer with ours
import difflib

# Split emojis
import functools
import operator
import emoji

# Non-contextual word embeddings
import gensim

# LSA
from sklearn.decomposition import TruncatedSVD

# For vector transofmation of tokens
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

# Model building
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB


# For model evaluation
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

###################################################################
##################### Load needed data ###########################
###################################################################
def loadLabels(filepath):

    """
    Load labels for given classification problem.
    :filepath: path to the file where are labels stored
    :return: list with labels as int
    """

    with open(filepath, "r") as f:
        labels = [int(label) for label in f.readlines()]
    
    return labels


###################################################################
##################### Preprocessing functions #####################
###################################################################
def expandContractions(line):

    """
    Used library: https://github.com/kootenpv/contractions

    Expands contractions.
    :line: string
    :return: string
    """

    # Get all words
    all_words = line.split(" ")

    # Fix the words
    result = []
    for word in all_words:
        expanded_word = contractions.fix(word)
        result.append(expanded_word)
    
    return " ".join(result)


def removeStopWords(words):

    """
    We got the idea which library to use from https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d
    (section "Stop-word removal")

    It removes english stopwords using nltk dataset.
    :words: list of words/tokens
    :return: Reduced list of words
    """

    # Get english stop words as a set
    english_sw = set(stopwords.words('english'))

    # Remove stopwords
    result = [word for word in words if word not in english_sw]

    return result


def normalizeUnicodeEmoji(dic):

    """
    We want to normalize description of emojis such that:
    1. There are no commas
    2. There are no colons
    Based on: https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d (See the below code comments for detail)
    :dic: Dictionary where key is the unicode of the given emoji and value is a text describing given emoji.
    :return: New dictionary with normalized values
    """

    # Here the thing which we used from the source is the idea that we should normalize
    # the emoji labels and the one line of code doing the normalization. But other than that, there is honestly not that much to be reinvented.
    # For complete reference use ctrl + f to search for: "Converting Emoji and Emoticons to words"
    result = dict()
    for emoji_code in dic:
        normalized_text = "_".join(dic[emoji_code].replace(",", "").replace(":", "").split())
        result[emoji_code] = normalized_text
    
    return result


def normalizeUnicodeEmoticons(dic):

    """
    We want to normalize description of emoticons such that:
    1. There are no commas
    Based on: https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d (See the below code comments for detail)
    :dic: Dictionary where key is the unicode of the given emoticon and value is a text describing given emoticon.
    :return: New dictionary with normalized values
    """

    # Here the thing which we used from the source is the idea that we should normalize
    # the emoticons labels and the one line of code doing the normalization. But other than that, there is honestly not that much to be reinvented.
    # For complete reference use ctrl + f to search for: "Converting Emoji and Emoticons to words"
    result = dict()
    for emoticon_code in dic:
        normalized_text = "_".join(dic[emoticon_code].replace(",", "").split())
        result[emoticon_code] = normalized_text
    
    # Add new emoticons (our own modification)
    result[":("] = "sad_face"
    result[":)"] = "smiley_face"
    result[":-)"] = "smiley_face"

    return result


def convertEmojis(tweet):

    """
    It turns emoji codes to words.
    Note: Could be improved using regex, however due to time constrainsts, we have not managed to do it.
    We got the idea of what libraries to use from this article: https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d
    (section "Converting Emoji and Emoticons to words")
    :tweet: String
    :return: string
    """

    # Get normalized emoji dic
    emoji_dic = normalizeUnicodeEmoji(UNICODE_EMO)
    adjusted_tweet = []
    for word in tweet.split():
        if word in emoji_dic:
            adjusted_tweet.append(emoji_dic[word])
        else:
            adjusted_tweet.append(word)
    
    return " ".join(adjusted_tweet)


def convertEmoticons(tweet):

    """
    It turns emoticons to words.
    Note: Could be improved using regex, however due to time constrainsts, we have not managed to do it.
    We got the idea of what libraries to use from this article: https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d
    (section "Converting Emoji and Emoticons to words")
    :tweet: String
    :return: string
    """

    # Get normalized emoticon dic
    emoti_dic = normalizeUnicodeEmoticons(EMOTICONS)
    adjusted_tweet = []
    for word in tweet.split():
        if word in emoti_dic:
            adjusted_tweet.append(emoti_dic[word])
        else:
            adjusted_tweet.append(word)
    
    return " ".join(adjusted_tweet)


def splitEmojis(tweet):

    """
    Based on the following SO question:
    https://stackoverflow.com/questions/49921720/how-to-split-emoji-from-each-other-python
    We used the Michael's Charemza answer. (first accepted answer)

    It solves the problem where you have multiple emojis right next to each other.
    :tweet: String
    :return: string
    """

    emojis_splitted = emoji.get_emoji_regexp().split(tweet)
    return " ".join(emojis_splitted)


###################################################################
##################### Preprocessing pipeline ######################
###################################################################
def prepareTweetForTokenizer(tweet, use_extended = True):

    """
    Prepares tweet for tokenizer. See the applied methods below.
    :tweet: String
    :use_extended: boolean, do you want to use extended version? (recommended)
    :return: string
    """

    # Escape html tags
    tweet = html.unescape(tweet) # &amp; --> &

    # Get rid of double encoding
    tweet = tweet.encode('raw_unicode_escape').decode('raw_unicode_escape') # \u+002c --> ,

    # Do also advanced things if specified
    if use_extended:

        # Split emojis: 😃😃 --> 😃 😃
        tweet = splitEmojis(tweet)

        # Turn emojis to words
        tweet = convertEmojis(tweet)
        
        # Turn emoticons to words, e.g. :-) --> smiley_face
        tweet = convertEmoticons(tweet)

        # Expand contractions, e.g. I'm --> I am
        tweet = expandContractions(tweet)

    # Lowercase
    tweet = tweet.lower()
    
    return tweet


def tokenizer(tweet, saveTo):

    """
    Tokenize tweets line by line and save it to the specified file.
    Strategy of tokeniser is based on the following demo code:
    https://learnit.itu.dk/pluginfile.php/295682/mod_resource/content/1/tok_demo.py (author: Christian Hardmeier)
    
    The core difference between the demo code and ourse is the part before "while tweet", i.e., regex patterns
    developed by our group.

    :tweet: string
    :saveTo: filepath
    :return: None
    """

    # Initialise list where you will save the outputs of tokenizer
    tokens = []
    unmatchable = []

    # Get token pattern to match
    # * Core
    core_pattern = "[a-z@#_’'$]+"
    core_pattern_contraction = "[a-z@#_]+[’'][a-z]+"
    core_pattern_advanced = "([a-z@#_’']+[-/&.][a-z@#_’']+)" # example: word1-word2
    # * Selected shortcuts
    shortcuts_pattern = "(w[/])"
    # * Final pattern
    token_pat = re.compile(rf"({core_pattern_contraction})|({core_pattern_advanced})|({shortcuts_pattern})|({core_pattern})|([!?,])|([.]+)|(…)")

    # Get patterns to skip
    # * Special characters
    spec_pat = '[()"]'
    skippable_pat = re.compile(rf'(\s+)|({spec_pat})')

    # Get rid of certain patterns (@user, url, numbers) before starting tweet analysis
    # Note, to see how the regex works, you can use https://regex101.com/ (as a delimiter use ;)
    url_pat = "(http[s]?://)([a-zA-Z]+)?[.]?[a-zA-Z0-9-]+[.][a-zA-Z-]+([a-zA-Z/0-9-%_.]+)+"

    # * Url pat, hint try:
    """
    We have spent 12 600 minutes on this project and we hope to get 12, or in worst case 11.99, or in
    percentage terms 100.00 %. Also note that deadline is 3rd of June and it is #endofspringsemester2021.
    """
    num_pat = "([0-9]+[a-z]+)|([0-9]+[ ][%])|([0-9]+[ ]*[0-9]+)|([0-9]+[.,]*[0-9]+[ ]*[%])|([0-9])|([a-z#]+[0-9])"

    # Use the above patterns
    tweet = re.sub(f"(@user)|({url_pat})|({num_pat})", "", tweet)

    # Iterate until there is nothing to be extracted from the tweet
    while tweet:
        
        # Try finding a skippable token delimiter first
        skippable_match = re.search(skippable_pat, tweet)

        # If there is one at the beginning of the line, just skip it
        if skippable_match and skippable_match.start() == 0:
            tweet = tweet[skippable_match.end():]
        
        # Otherwiese try finding a real token based on the pattern above
        else:

            # Try to get the match
            token_match = re.search(token_pat, tweet)

            # If there is a token at the start of the line, tokenise it
            if token_match and token_match.start() == 0:
                tokens.append(tweet[:token_match.end()])
                tweet = tweet[token_match.end():]
            
            # Otherwise we encountered either of the two following scenarios:
            # 1. It ends where a skippable or token match starts
            # 2. It ends at the end of the line
            else:
                unmatchable_end = len(tweet)
                if skippable_match:
                    unmatchable_end = skippable_match.start()
                if token_match:
                    unmatchable_end = min(unmatchable_end, token_match.start())

                # Update unmatchable list and adance in the tweet processing
                unmatchable.append(tweet[:unmatchable_end])
                tweet = tweet[unmatchable_end:]
    
    # Remove stopwords
    tokens = removeStopWords(tokens)
    
    # Save to csv csv or show the result
    if saveTo == "print":
        print("Matched tokens: ", tokens)
        print("Unmatched tokens: ", unmatchable)
    else:
        with open(saveTo, "a") as f:
            f.write(" ".join(tokens) + "\n")


def tokenizeTweets(filepath, saveTo, use_extended = True, skip_lines = 0):

    """
    Wrapper function around tokeniser.
    :filepath: path to raw data
    :saveTo: path to folder where to save data
    :use_extended: Do you want to use the extended version of tokeniser (recommended)
    :skip_lines: do you want to skip certain number of lines
    :return: None
    """

    with open(filepath, encoding = "utf-8") as f:

        # Read first line
        tweet = f.readline()

        # Iterate over tweets and tokenise them
        while tweet:
            
            if skip_lines == 0:

                # Prepare
                tweet = prepareTweetForTokenizer(tweet, use_extended = use_extended)

                # Tokenise given tweet and save it to the specified file
                tokenizer(tweet, saveTo)

            else:
                skip_lines -= 1

            # Read next line
            tweet = f.readline()


###################################################################
##################### Evaluation of tokenizer #####################
###################################################################
def evaluateAgainstNltk(filepaths, saveTo, compareTo, skip_lines = 0):

    """
    Compares our tokeniser with NLTK tokeniser using quantitative method: SequenceMatcher ratio metric.
    :filepaths: iterbale which has path to training and validation dataset
    :saveTo: where to save output of nltk tokenizer
    :compareTo: filepath to tokenised Training dataset
    :skip_lines: how many lines do you want to skip when tokenizing data
    :return: list with sequence ratio scores
    """

    # Prepare tokenizer
    tknzr = TweetTokenizer()

    # Make sure there is no file within the folder (we are using append method, so we want to append to empty file)
    file_to_delete = glob.glob(saveTo)
    if file_to_delete:
        os.remove(file_to_delete[0])

    for filepath in filepaths:
        with open(filepath) as f:

            # Read first line
            tweet = f.readline()

            # Iterate over tweets and tokenise them
            while tweet:
                
                if skip_lines == 0:
                    
                    # Tokenise given tweet and save it to the specified file
                    tokens = tknzr.tokenize(tweet)

                    # Save the result
                    with open(saveTo, "a") as file_save:
                        file_save.write(" ".join(tokens) + "\n")

                else:
                    skip_lines -= 1

                # Read next line
                tweet = f.readline()

    # Compare it
    nltk_extended_ratio = []
    with open(compareTo) as f1, open(saveTo) as f2:

        # Read first line
        a = f1.readline()
        b = f2.readline()

        while a and b:
            seq = difflib.SequenceMatcher(None, a.strip("\n"), b.strip("\n"), autojunk=False)
            nltk_extended_ratio.append(seq.ratio())
            a = f1.readline()
            b = f2.readline()
    
    return nltk_extended_ratio



def getTokenDeltas(filepath1, filepath2, r):
    '''
    This function prints a human-readable text for comparing two strings.
    Differences are marked using the Python difflib' function 'ndiff'.
    :filepath1:, :filepath2: filepaths to lists of strings.
    :r: number of sentences to compare
    '''
    with open(filepath1) as s1, open(filepath2) as s2:
        for _ in range(r):
            # Read first lines in range 'r' of both data sets
            line = s1.readline().replace(',',' ') # read words as space seperated
            line_nltk = s2.readline().replace(',',' ') # read words as space seperated

            # Print readble differences using ndiff function from difflib library
            print("\n".join(difflib.ndiff([line], [line_nltk])))


###################################################################
####################### CHARACTERIZING DATA #######################
###################################################################
def build_corpus_freq(filepath):
    ''' 
    The strategy for corpus analysis is based on the following demo code:
    https://learnit.itu.dk/pluginfile.php/296257/mod_resource/content/1/zipf.py (author: Christian Hardmeier)
    
    This function reads a tokenized .csv file and creates a corpus and vocabulary for the specific
    file. It then adds the vocabulary in descendin order to a pandas dataframe togehter with a 
    normalized and cumulative frequency count.

    :filepath: path to tokenized data

    :return: corpus (complete list of tokens), vocabulary (dict. of <token, freq.> paris)
        and a frequency-ordered dataframe containing all tokens and frequencies.
    '''
    corpus = []
    # Read one line (tweet) at at time and add tokens to corpus list
    with open(filepath, 'r') as f:
        for line in f:
            to_list = line.rstrip().split(" ") # split tokens by space
            corpus.extend(t for t in to_list)

    voc = collections.Counter(corpus)
    frq = pd.DataFrame(voc.most_common(), columns=['token', 'frequency'])

    # Index in the sorted list
    frq['idx'] = frq.index + 1

    # Frequency normalised by corpus size
    frq['norm_freq'] = frq.frequency / len(corpus)

    # Cumulative normalised frequency
    frq['cumul_frq'] = frq.norm_freq.cumsum()

    # Log frequency
    frq['log_frq'] = np.log(frq.frequency)

    # Log rank (in descending order)
    frq['log_rank'] = np.log(frq.frequency.rank(ascending=False))

    return corpus, voc, frq

def cor_voc_size(corpus, voc):
    '''
    This function returns the corpus and vocabulary 
    size together with the token / type ratio.
    :corpus: list of tokens

    '''
    # Corpus size
    print("Corpus size:", len(corpus))

    # Vocabulary size
    print("Vocabulary size:", len(voc))

    # Token / type ratio
    print("Type / Token Ratio:", round((len(voc) / len(corpus)),3))


###################################################################
####################### ANNOTATION ANALYSIS #######################
###################################################################

def cohen_kappa(ann1, ann2):
    '''
    Code inspirered from: https://towardsdatascience.com/inter-annotator-agreement-2f46c6d37bf3 
    (see code snippet under "Cohen Kappa"). This follows the general equation for computing 
    Cohens Kappa.
    
    This function computes Cohen Kappa for pair-wise annotators.
    :ann1: list of annotations provided by first annotator 
    :ann2: list of annotations provided by second annotator
    :return: Cohen's Kappa score
    '''
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (P_0)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (P_e)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count
    
    # return Cohen's Kappa
    return round((A - E) / (1 - E), 3)



def fleiss_kappa(M):
    '''
    Code inspirered from: https://towardsdatascience.com/inter-annotator-agreement-2f46c6d37bf3 
    (see code snippet under "Fleiss' Kappa"). This follows the general equation for computing 
    Fleiss' Kappa.

    This function computes Fleiss' Kappa for a group of two or more annotators.
    :M: a numpy matrix of shape (N,k) with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :return: Fleiss' Kappa score
    '''
    N, k = M.shape  # N is no. of items, k is no. of categories
    n_annotators = float(np.sum(M.iloc[0, :]))  # no. of annotators
    tot_annotations = N * n_annotators  # the total no. of annotations
    category_sum = np.sum(M, axis=0)  # sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)


###################################################################
######################## Feature extraction #######################
###################################################################
class streamTokens:

    """
    We were tasked with the problem of implementing
    "restartable iterable". This term came up when we wanted to load data
    into gensim.models.Word2Vec and the author said that it is not enough
    to have a generator function, but also that the iterable must be restartable.
    Not being sure what it means, we found the following resource which is in fact
    written by the co-author of the gensim library:
    https://rare-technologies.com/data-streaming-in-python-generators-ite (author: Radim Řehůřek)

    The general idea what is meant by the restartable iterable is explained
    within the article. Most importanlty we took inspiration and based the
    implementation of this class on the class in the article which you can find
    using ctrl + f "class TxtSubdirsCorpus". Note, however, that the main thing
    we used is the idea of what is meant by "restartable iterable", but the below implementation
    is ours.
    """
    
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.N = self.getLen()
 
    def __iter__(self):

        """
        Implementing a generator.
        """
        
        for filepath in self.filepaths:
            with open(filepath, "r", encoding='utf-8') as f:

                raw_tweet = f.readline()

                while raw_tweet:

                    # Get the actual tweet
                    tweet = raw_tweet.strip("\n").split(" ") # ["Example", "of", "tokenized", "tweet"]

                    # Update raw tweet
                    raw_tweet = f.readline()
                    
                    yield tweet
        
    def getLen(self):

        """
        From https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
        I confirmed my intuition that the most efficient way to (at least in terms of memory) get number of rows in the given file,
        is to iterate over all lines one at a time and count them.
        :return: Number of lines within specified files within the construction function.
        """

        count = 0
        for filepath in self.filepaths:
            with open(filepath) as f:
                for l in f:
                    count += 1

        return count
    
    def __len__(self):
        return self.N

def return_tweet(tweet):

    """
    Create dummy function representing tokenizer.
    You can view this as an "identity" function as you suggested in last lecture
    for solving this problem.
    :return: what you were given, tweet.
    """

    return tweet


def getWordVector(tokens, size, model_w2v):

    """
    Source:
    https://www.kaggle.com/nitin194/twitter-sentiment-analysis-word2vec-doc2vec

    See below comments for each piece of code to know what is our work
    and what was used.
    :tokens: iterable with tokens
    :size: number of features, int
    :model_w2v: gensim.models.Word2Vec object
    :return: Corresponding word vector as 1D numpy array
    """

    # Create a vector
    # This piece of code was completely copied
    # For reference, see: ctrl + f "In [35]"
    vector = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:

        # Handle cases where the token is not in vocabulary
        try:
            vector += model_w2v.wv[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vector /= count

    return vector

def getWord2VecMatrix(tokenized_tweets, vector_size):

    """
    Source:
    https://www.kaggle.com/nitin194/twitter-sentiment-analysis-word2vec-doc2vec

    See below comments for each piece of code to know what is our work
    and what was used.
    :tokenized_tweets: iterable
    :vector_size: Number of features
    :return: 2D numpy array as a matrix
    """

    # Build model
    # Completely copied including the parameters
    model_w2v = gensim.models.Word2Vec(
            tokenized_tweets,
            vector_size=vector_size,
            window=5,
            min_count=2,                           
            sg = 1, 
            hs = 0,
            negative = 10,
            workers= 32,
            seed = 34
            )
    
    # Train model on the provided data
    # Completely copied, but here I would argue there is not that match to be reinvented
    model_w2v.train(tokenized_tweets, total_examples= len(tokenized_tweets), epochs=20)

    # Build the matrix and save it
    # Here we also copied the code, but to be honest there is no other
    # way how you can save vector rows to matrix
    # For exact comparison, go to link above and ctrl + f "In [36]:"
    word_vectors = np.zeros((len(tokenized_tweets), vector_size))
    for i, tokenized_tweet in enumerate(tokenized_tweets):
        word_vectors[i,:] = getWordVector(tokenized_tweet, vector_size, model_w2v)
    
    return word_vectors


def vectorizeTokensToMatrix(tweets, method = "CountVectorizer"):

    """
    Turn the given set of tweets into features using the specified method.
    :tweets: iterable
    :method: CountVectorizer, TfidfVectorizer, HashingVectorizer, Word2Vec, LSA
    :return: matrix with features (different implementations: 2d numpy array, sklearn sparse matrix)
    """


    # Initiliaze the vectorizer
    if method == "CountVectorizer":
        vectorizer = CountVectorizer(
                tokenizer=return_tweet,
                preprocessor=return_tweet,
                token_pattern=None,
                ngram_range=(1, 3)
        )
    elif method == "TfidfVectorizer":
        vectorizer = TfidfVectorizer(
                tokenizer=return_tweet,
                preprocessor=return_tweet,
                token_pattern=None,
                ngram_range=(1, 2)
        )
    elif method == "HashingVectorizer":
        vectorizer = HashingVectorizer(
                tokenizer=return_tweet,
                preprocessor=return_tweet,
                token_pattern=None,
                ngram_range=(1, 2),
                n_features=2**20
        )


    # Get matrix
    # Word2Vec
    if method == "Word2Vec":
        matrix = getWord2VecMatrix(tweets, vector_size = 200)

    # LSA
    elif method == "LSA":

        # Use tf-idf 
        vectorizer = TfidfVectorizer(
                tokenizer=return_tweet,
                preprocessor=return_tweet,
                token_pattern=None,
                ngram_range=(1, 2)
        )

        # Get the corresponding sparse matrix
        tf_idf_matrix = vectorizer.fit_transform(tweets)

        # Perform LSA
        svd = TruncatedSVD(n_components=100, random_state=42)
        matrix = svd.fit_transform(tf_idf_matrix)

    # CountVectorizer, TfidfVectorizer, HashingVectorizer
    else:
        matrix = vectorizer.fit_transform(tweets)

    return matrix


###################################################################
############################ Classifiers ##########################
###################################################################
def loadModelData(whichData, FILEPATHS, method = "CountVectorizer", skip_lines = 0):

    """
    Returns training and testing data using the given feature extraction method.
    :whichData: 'hate' or 'sentiment'
    :FILEPATHS: dict
    :skip_lines: skip certain number of lines from the training data
    :return: X_train, X_test, y_train, y_test
    """
    

    # GET TRAINING AND TESTING DATA
    # * Get features based on specified method
    filepaths = [FILEPATHS[f"save_train_{whichData}"], FILEPATHS[f"save_test_{whichData}"]]
    features = vectorizeTokensToMatrix(
        tweets = streamTokens(filepaths),
        method = method
    )

    # * Get number of rows for training
    N = streamTokens([filepaths[0]]).N # Len of training

    # * Create training and testing dataset
    X_train = features[:N]
    X_test = features[N:]

    # GET LABELS
    y_train = loadLabels(FILEPATHS[f"train_labels_{whichData}"])[skip_lines:] + loadLabels(FILEPATHS[f"validate_labels_{whichData}"])
    y_test = loadLabels(FILEPATHS[f"test_labels_{whichData}"])

    return X_train, X_test, y_train, y_test


def getConfusionMatrix(X_train, y_train, estimators, cv):

    """
    First it computes confusion matrix for each Kth fold.
    Second, it condenses all the K confusion matrices into one using mean.
    :estimators: List with models which were fitted using K-th training data
    :cv: StratifiedShuffleSplit object which is used to obtain validation dataset
    :return: confusion matrix as ndarray
    """
    
    confusion_matrices = []
    iterator = 0
    for _, test_index in cv.split(X_train, y_train):

        # Get the validation datasets for Kth fold
        X_validate = X_train[test_index]
        y_validate =  np.array(y_train)[test_index]

        # Make a prediction using the fitted model
        prediction = estimators[iterator].predict(X_validate)

        # Compute the confusion matrix
        cm = confusion_matrix(y_true = y_validate, y_pred = prediction)
        confusion_matrices.append(cm)

        iterator += 1
    
    # Compute the average
    result = sum(confusion_matrices)/len(confusion_matrices)

    return result

def getSummaryOfBinaryCrossValRes(X_train, y_train, clfs, metrics, cv):

    """
    Returns a summary for the given classifiers and metrics.
    :X_train: sparse matrix
    :y_train: 1d array like object
    :clfs: dict, where key is name of the classifier, and value is the instance with concrete parameters of the given classifier
    :metrics: list with names of the metrics, see documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
    :cv: StratifiedShuffleSplit object with given splits
    :return: Summary as df where rows are classifier and columns are metrics, confusion matrix for each classifier
    """

    # Build a dataframe where you will save the results of cross-validation
    header = ["classifier_name"] + metrics
    # * Build the empty dataframe
    results = pd.DataFrame(columns = header)

    # Compute the results
    all_cms = dict() # Here you store all confusion matrices for each classifier

    for name, clf in clfs.items():

        # Get the results for each metric as dict
        result_dict = cross_validate(clf, X_train, y_train, cv=cv, scoring = metrics, return_estimator = True)

        # Condense the results using their mean and save it to the list
        result = []
        for metric_name in metrics:
            key =  f"test_{metric_name}"
            value = np.mean(result_dict[key])
            result.append(value)
        
        # Get estimators produced within the K-folds and use them to get confusion matrix
        estimators = result_dict["estimator"]

        # Compute confusion matrix and add it to the dictionary
        cm = getConfusionMatrix(X_train, y_train, estimators, cv)
        all_cms[name] = cm
        
        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        results = results.append(new_record, ignore_index=True)
    
    return results, all_cms


def customMultiClassCrossVal(X, y, clf, metrics, cv, labels):

    """
    This class implements custom Multiclass Cross validation. This is mainly
    due to the fact that we want to be able to see precision, recall and f1 score
    for individual classes.

    :X: sparse matrix
    :y: 1d array like object
    :clf: concrete instance of an classifier
    :metrics: list with names of the metrics, see documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
    You can select among these: preccision, recall, f1
    :cv: StratifiedShuffleSplit object with given splits
    :labels: String labels for each particular class
    :return: summary_df where rows are metrics and columns are given labels, confusion matrix
    """

    # Transform y to an array
    y = np.array(y)

    # Build a dataframe where you will save the results of cross-validation
    header = ["Metrics/Classes"] + labels
    # * Build the empty dataframe
    results = pd.DataFrame(columns = header)

    # Also save confusion matrices for each K-fold
    confusion_matrices = []

    # Do K runs
    for train_index, test_index in cv.split(X, y):

        # Get the split: training and validation data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the model using the training data
        clf.fit(X_train, y_train)

        # Make predictions
        predictions = clf.predict(X_test)

        # Evaluate using the specified metrics
        for metric in metrics:
            
            # Create a new record
            if "precision" == metric:
                precision_scores = precision_score(y_true = y_test, y_pred = predictions, average = None, labels = [0, 1, 2])
                new_record = pd.DataFrame([[metric] + precision_scores.tolist()], columns = header)
            
            if "recall" == metric:
                recall_scores = recall_score(y_true = y_test, y_pred = predictions, average = None, labels = [0, 1, 2])
                new_record = pd.DataFrame([[metric] + recall_scores.tolist()], columns = header)
            
            if "f1" == metric:
                f1_scores = f1_score(y_true = y_test, y_pred = predictions, average = None, labels = [0, 1, 2])
                new_record = pd.DataFrame([[metric] + f1_scores.tolist()], columns = header)
            
            # Add it as a result to the results table
            results = results.append(new_record, ignore_index=True)

        # Compute confusion matrix
        cm = confusion_matrix(y_true = y_test, y_pred = predictions)
        confusion_matrices.append(cm)
    
    # Compute the final metrics using average
    final_results = pd.DataFrame(columns = header)
    for metric in metrics:

        # Select only rows where the given metric is
        rows_with_metric = results[results["Metrics/Classes"] == metric]

        # Compute the mean along columns
        # If you are confused about which axis to use, see: https://stackoverflow.com/questions/25773245/ambiguity-in-pandas-dataframe-numpy-array-axis-definition
        # Use axis=0 to apply a method down each column
        # For each metric, this will return its average from K splits
        final_metric = pd.DataFrame.mean(rows_with_metric.iloc[:, 1:], axis = 0) 

        # Add it to the dataframe
        new_record = pd.DataFrame([[metric] + final_metric.tolist()], columns = header)
        final_results = final_results.append(new_record, ignore_index=True)

    # Compute the final confusion matrix
    confusion_matrix_final = sum(confusion_matrices)/len(confusion_matrices)
    
    return final_results, confusion_matrix_final


def getSummaryOfMulticlassCrossValRes(X, y, clfs, metrics, cv):

    """
    Compute the summary for given classifiers and metrics along with corresponding confusion matrix.
    :X: sparse matrix
    :y: 1d array like object
    :clfs: dict, where key is name of the classifier, and value is the instance with concrete parameters of the given classifier
    :metrics: list with names of the metrics, see documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
    You can select among these: preccision, recall, f1
    :cv: StratifiedShuffleSplit object with given splits
    :return: dictionary: classifier_name: (summary_df, confusion matrix), overall metrics (accuracy, f1_macro) summary
    """

    # Compute the results for individual metrics (recall, precisssion, f1)
    # Save the final result as a dictionary: classifier_name: (summary_df, confusion matrix)
    final_result = dict()

    # Compute the results
    for name, clf in clfs.items(): 

        # Do the cross validation
        labels = ["Negative", "Neutral", "Positive"]
        classifier_summary, confusion_matrix_final = customMultiClassCrossVal(X, y, clf, metrics, cv, labels)

        # Save the results
        final_result[name] = (classifier_summary, confusion_matrix_final)
    
    # Compute the results for overall metrics
    # Define scoring
    scoring = ["accuracy", "f1_macro", "recall_macro"]
    # Build a dataframe where you will save the results of cross-validation
    header = ["Classiffier"] + scoring
    # * Build the empty dataframe
    overall_metrics = pd.DataFrame(columns = header)

    for name, clf in clfs.items():

        result_dict = cross_validate(clf, X, y, cv=cv, scoring = scoring)

        # Condense the results using their mean and save it to the list
        result = []
        for metric_name in scoring:
            key =  f"test_{metric_name}"
            value = np.mean(result_dict[key])
            result.append(value)
        
        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        overall_metrics = overall_metrics.append(new_record, ignore_index=True)

    return final_result, overall_metrics


def crossValidate(X_train, y_train, clfs, metrics, binary = True):

    """
    Run cross validation on the given input. Use StratifiedShuffleSplit to make the splits
    such that balance between labels is preserved.
    :X_train: sparse matrix
    :y_train: 1d array like object
    :clfs: dict, where key is name of the classifier, and value is the instance with concrete parameters of the given classifier
    :metrics: list with names of the metrics, see documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
    :binary: Is the classification binary or multiclass?
    :return: See the detail within the corresponding function: binary vs multiclass
    """

    # Prepare cross-validation
    # * Specify K - industry standard is 5 - 10
    K = 5
    cv = StratifiedShuffleSplit(n_splits=K, test_size=0.2, random_state=1)

    # Return corresponding summary: Binary or Multiclass
    if binary:
        return getSummaryOfBinaryCrossValRes(X_train, y_train, clfs, metrics, cv)
    else:
        return getSummaryOfMulticlassCrossValRes(X_train, y_train, clfs, metrics, cv)



###################################################################
###################### Evaluation of models #######################
###################################################################
def fitClassifiers(clfs, X_train, y_train):

    """
    Use training data to fit specified classifiers and return the
    result back as a dict.
    :clfs: dictionary
    :X_train: sparse matrix
    :y_train: 1d array like object
    :return: dict where key is the name of classifier and value is fitted classifier.
    """

    result = dict()
    for clf_name, clf in clfs.items():
        fitted_model = clf.fit(X_train, y_train)
        result[clf_name] = fitted_model
    
    return result


def evaluateTestDataBinary(X_test, y_true, clfs):

    """
    The goal of this function is to evaluate performance of
    selected classifiers on Test data.
    :X_test: sparse matrix
    :y_true: 1d array like object
    :clfs: dict where key is the name of classifier and value is fitted classifier.
    :return: summary df where row is name of the classifier, columns are selected metrics
    """

    # Build a dataframe where you will save the results
    metrics = ["accuracy", "precision", "recall", "f1", "f1_macro"]
    header = ["classifier_name"] + metrics
    results = pd.DataFrame(columns = header)

    # Compute the results
    for name, clf in clfs.items():
        
        # Make preddictins
        y_pred = clf.predict(X_test)

        # Compute the metrics
        # * Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # * Precision
        prec = precision_score(y_true, y_pred)

        # * Recall
        rec = recall_score(y_true, y_pred)

        # * F1
        f1 = f1_score(y_true, y_pred)

        # * F1 macro
        f1_macro = f1_score(y_true, y_pred, average = "macro")

        result = [acc, prec, rec, f1, f1_macro]

        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        results = results.append(new_record, ignore_index=True)
    
    return results


def getOverallStatsMultiClassEval(X_test, y_true, clfs):

    """
    :X_test: sparse matrix
    :y_true: 1d array like object
    :clfs: dict where key is the name of classifier and value is fitted classifier.
    :return: summary df where row is name of the classifier, columns are selected overall metrics
    """

    # Build a dataframe where you will save the results
    metrics = ["accuracy",  "f1_macro", "recall_macro"]
    header = ["classifier_name"] + metrics
    results = pd.DataFrame(columns = header)

    # Compute the results
    for name, clf in clfs.items():
        
        # Make preidctions
        y_pred = clf.predict(X_test)

        # Compute the metrics
        # * Accuracy
        acc = accuracy_score(y_true, y_pred)

        # * F1 macro
        f1_macro = f1_score(y_true, y_pred, average = "macro")

        # * Recall macro
        rec_macro = recall_score(y_true, y_pred, average = "macro")

        result = [acc, f1_macro, rec_macro]

        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        results = results.append(new_record, ignore_index=True)
    
    return results


def evaluateTestDataMulti(X_test, y_test, clfs):

    """
     The goal of this function is to evaluate performance of
    selected classifiers on Test data where we solve Multiclass problem.
    :X_test: sparse matrix
    :y_test: 1d array like object
    :clfs: dict where key is the name of classifier and value is fitted classifier.
    :return: 
    - dict where key is the name of classifier and value is a df where rows are metrics and columns classess.
    - summary df where row is name of the classifier, columns are selected overall metrics
    """

    # Save results to a dictionary such that: key: clf_name value: summary_df
    clfs_results = dict()

    # Compute the results for "precision", "recall", "f1" and given labels per classifier
    for name, clf in clfs.items():

        # Make predictions
        predictions = clf.predict(X_test)

        # Build a dataframe where you will save the results
        header = ["Metrics/Classes"] + ["Negative", "Neutral", "Positive"]
        results = pd.DataFrame(columns = header)

        # Evaluate using the specified metrics
        metrics = ["precision", "recall", "f1"]
        for metric in metrics:
            
            # Create a new record
            if "precision" == metric:
                precision_scores = precision_score(y_true = y_test, y_pred = predictions, average = None, labels = [0, 1, 2])
                new_record = pd.DataFrame([[metric] + precision_scores.tolist()], columns = header)
            
            if "recall" == metric:
                recall_scores = recall_score(y_true = y_test, y_pred = predictions, average = None, labels = [0, 1, 2])
                new_record = pd.DataFrame([[metric] + recall_scores.tolist()], columns = header)
            
            if "f1" == metric:
                f1_scores = f1_score(y_true = y_test, y_pred = predictions, average = None, labels = [0, 1, 2])
                new_record = pd.DataFrame([[metric] + f1_scores.tolist()], columns = header)
            
            # Add it as a result to the results table
            results = results.append(new_record, ignore_index=True)
        
        # Save the results for given classifier into a dict
        clfs_results[name] = results
    
    # Compute the overall statistics
    overall_stats = getOverallStatsMultiClassEval(X_test, y_test, clfs)
    
    return clfs_results, overall_stats
