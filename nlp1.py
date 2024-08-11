import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the stopwords
nltk.download('stopwords')
nltk.download('punkt')

text = "This is an example sentence demonstrating stop word removal."

# Tokenize the text
words = word_tokenize(text)

# Get the list of stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]

print("Original:", words)
print("Filtered:", filtered_words)

import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

text = "This is an example sentence demonstrating stop word removal."

# Process the text
doc = nlp(text)

# Remove stopwords
filtered_words = [token.text for token in doc if not token.is_stop]

print("Original:", [token.text for token in doc])
print("Filtered:", filtered_words)

from gensim.parsing.preprocessing import remove_stopwords

text = "This is an example sentence demonstrating stop word removal."

# Remove stopwords
filtered_text = remove_stopwords(text)

print("Original:", text)
print("Filtered:", filtered_text)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

text = "This is an example sentence demonstrating stop word removal."

# Tokenize the text
words = text.split()

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]

print("Original:", words)
print("Filtered:", filtered_words)

