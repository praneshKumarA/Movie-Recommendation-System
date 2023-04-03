import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import spacy
from utils import return_df
from utils import return_stop_words

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in return_stop_words()] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(' '.join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def Sort_Tuple(tup):  
    return(sorted(tup, key = lambda x: x[1], reverse = True))

def main():
    df = return_df()
    df['Description'] = df['Description'].map(lambda x: re.sub('([^\x00-\x7F])+','', x))
    data_words = list(sent_to_words(df['Description']))
    
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10) 
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    data_words_nostops = remove_stopwords(data_words)

    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    data_words_nostops = remove_stopwords(data_words)

    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    id2word = corpora.Dictionary(data_lemmatized)
    
    id2word.filter_extremes(no_below=2, no_above=0.9)

    texts = data_lemmatized

    corpus = [id2word.doc2bow(text) for text in texts]
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=15, 
                                        random_state=100,
                                        chunksize=100,
                                        passes=10,
                                        alpha=0.01,
                                        eta=0.9)
    doc_lda = lda_model[corpus]
    
    doc_num, topic_num, prob = [], [], []
    for n in range(len(df)):
        get_document_topics = lda_model.get_document_topics(corpus[n])
        doc_num.append(n)
        sorted_doc_topics = Sort_Tuple(get_document_topics)
        topic_num.append(sorted_doc_topics[0][0])
        prob.append(sorted_doc_topics[0][1])
    df['Doc'] = doc_num
    df['Topic'] = topic_num
    df['Probability'] = prob
    df.to_csv("doc_topic_matrix.csv", index=False)
    