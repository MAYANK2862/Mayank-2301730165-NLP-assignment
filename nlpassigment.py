# ===============================
# NLP Assignment Pipeline
# ===============================

# Import necessary libraries
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Step 0: Custom Paragraph
paragraph = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction
between computers and humans through natural language. Real-time applications include chatbots, 
voice assistants, and sentiment analysis.
"""

print("Original Paragraph:")
print(paragraph)

# Step 1: Tokenization using TreebankWordTokenizer (avoids punkt issues)
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(paragraph)
print("\nStep 1 - Tokenization:")
print(tokens)

# Step 2: Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\nStep 2 - Stopword Removal:")
print(filtered_tokens)

# Step 3: Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\nStep 3 - Stemming:")
print(stemmed_tokens)

# Step 4: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nStep 4 - Lemmatization:")
print(lemmatized_tokens)




#NLP Definition and Application

'''nlp_definition = 
Natural Language Processing (NLP) is a subfield of Artificial Intelligence (AI) 
that enables computers to understand, interpret, and generate human language.
"""

nlp_application = """
Real-time Application (Healthcare Domain):
- NLP can analyze patient records, extract important medical information, 
  detect disease patterns, and assist in clinical decision-making.
- Example: AI-powered chatbots for virtual health consultations, or 
  extracting symptoms from medical notes automatically.
"""

print("\n--- NLP Definition ---")
print(nlp_definition)
print("\n--- NLP Real-time Application (Healthcare) ---")
print(nlp_application)'''


# NLU and NLG Definitions

'''nlu_definition = 
Natural Language Understanding (NLU) is a branch of NLP that focuses on 
understanding the meaning of text or speech. 
Example: Determining sentiment from a customer review or understanding commands in a voice assistant.
"""

nlg_definition = """
Natural Language Generation (NLG) is a branch of NLP that focuses on 
generating human-like text from structured data. 
Example: Automatically creating weather reports, news summaries, or chatbot responses.
"""

print("\n--- NLU Definition ---")
print(nlu_definition)
print("\n--- NLG Definition ---")
print(nlg_definition)'''
