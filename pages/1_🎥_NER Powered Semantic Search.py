import streamlit as st

import utils
import embedding

df = embedding.return_df()
pinecone = embedding.pinecone_index()
index_name = 'ner-search'
if index_name not in pinecone.list_indexes():
    index = embedding.create_pinecone_index(pinecone, index_name)
    embedding.upsert_vectors(index)
else:
    index = pinecone.Index(index_name)

def search_pinecone(movie_name):
    df_query = df[df['Title'].apply(lambda x: x.lower()) == movie_name.lower()].reset_index(drop = True)
    if df_query.empty:
        return None
    query = df_query['Description'][0]
    query_punc_removed = df_query['Description_punc_removed'][0]
    ne = utils.extract_named_entities([query])[0]
    xq = utils.return_retriever().encode(query_punc_removed).tolist()
    xc = index.query(xq, top_k = 5, include_metadata = True, filter = {"named_entities" : {"$in" : ne}})
    movie = [(x['metadata']['Title'], x['metadata']['Description']) for x in xc['matches'] if x['metadata']['Title'].lower() != movie_name.lower()]
    return movie

st.set_page_config(page_title = 'Content Based Recommendation using NER powered Semantic Search', page_icon = '🎥')
st.markdown('# Movie Recommendation System')
text_input_1 = st.text_input(
    "Enter a Movie / Series Name 👇", 
    label_visibility = 'visible', 
    placeholder = 'Movie / Series', 
    key = 'ner'
)
if text_input_1:
    with st.spinner('Loading...'):
        result = search_pinecone(text_input_1)
    if result is None:
        st.warning('Please enter a valid movie / series name', icon = '⚠️')
    elif len(result) == 0:
        st.warning('No similar movie / series is found', icon = '❌')
    else:
        for i in result:
            with st.expander(i[0]):
                st.write(i[1])
else:
    st.warning('Please enter a movie / series name', icon = '⚠️')
