import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import string

device = torch.cuda.current_device() if torch.cuda.is_available() else None

model_id = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(
    model_id
)

model = AutoModelForTokenClassification.from_pretrained(
    model_id
)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="max",
    device=device
)

def return_retriever():
    retriever = SentenceTransformer(
        'flax-sentence-embeddings/all_datasets_v3_mpnet-base',
        device=device
    )
    return retriever

def extract_named_entities(text_batch):
    extracted_batch = ner_pipeline(text_batch)
    entities = []
    for text in extracted_batch:
        ne = [entity["word"] for entity in text]
        entities.append(ne)
    return entities

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower()

def return_df():
    df = pd.read_csv('netflix_titles.csv', index_col = [0])

    df['Description_punc_removed'] = df['Description'].apply(remove_punctuation)

    df['Title'] = df['Title'].apply(lambda x: x[x.index('.') + 2:-7] if (x[-1] == ')' and x[-6] == '(') and (x[1] == '.' or x[2] == '.' or x[3] == '.' or x[4] == '.') else x)
    
    return df

stop_words = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]

def return_stop_words():
    return stop_words
