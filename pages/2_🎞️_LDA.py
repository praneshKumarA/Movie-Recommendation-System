import streamlit as st
import LDA
import pandas as pd
from os.path import exists

file_name = 'doc_topic_matrix.csv'
file_exists = exists(file_name)

if file_exists:
    df = pd.read_csv(file_name)
else:
    LDA.main()
    df = pd.read_csv(file_name)
    
def recommend_by_storyline(title, df):
    recommended = []
    top10_list = []
    
    if df[df['title'].apply(lambda x: x.lower()) == title.lower()].reset_index(drop = True).empty:
        return None
    topic_num = df[df['title'].apply(lambda x: x.lower()) == title.lower()].Topic.values
    doc_num = df[df['title'].apply(lambda x: x.lower()) == title.lower()].Doc.values    
    
    output_df = df[df['Topic']==topic_num[0]].sort_values('Probability', ascending=False).reset_index(drop=True)

    index = output_df[output_df['Doc']==doc_num[0]].index[0]
    
    top10_list += list(output_df.iloc[index-5:index].index)
    top10_list += list(output_df.iloc[index+1:index+6].index)
    
    output_df['title'] = output_df['title'].str.title()
    
    for each in top10_list:
        recommended.append((output_df.iloc[each].title, output_df.iloc[each].description))
        
    return recommended

st.set_page_config(page_title = 'Content Based Recommendation using LDA', page_icon = '🎥')
st.markdown('# Movie Recommendation System')
text_input_2 = st.text_input(
    "Enter a Movie / Series Name 👇", 
    label_visibility = 'visible', 
    placeholder = 'Movie / Series', 
    key = 'lda'
)
if text_input_2:
    with st.spinner('Loading...'):
        result = recommend_by_storyline(text_input_2, df)
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


