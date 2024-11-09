import pinecone
from tqdm.auto import tqdm

import utils
from utils import return_df

def pinecone_index():
    pinecone = Pinecone(api_key="ec1a3181-1a35-4441-a871-c06df4567b0d")
    # pinecone.init(
    #     api_key="ec1a3181-1a35-4441-a871-c06df4567b0d",
    #     environment="eu-west1-gcp"
    # )


    return pinecone

def create_pinecone_index(pinecone, index_name):
    pinecone.create_index(
        index_name, 
        dimension = 768, 
        metric = 'cosine'
    )
    index = pinecone.Index(index_name)
    return index
    

def upsert_vectors(index):

    df = return_df()

    batch_size = 64

    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i+batch_size, len(df))
        batch = df.iloc[i:i_end]
        emb = utils.return_retriever().encode(batch["Description_punc_removed"].tolist()).tolist()
        batch_text = batch['Description'].tolist()
        entities = utils.extract_named_entities(batch_text)
        batch["named_entities"] = [list(set(entity)) for entity in entities]
        batch = batch.drop(['type', 'director', 'cast', 'country', 'date_added' ,'release_year', 'rating', 'duration', 'listed_in'], axis=1)
        meta = batch.to_dict(orient="records")
        ids = [f"{idx}" for idx in range(i, i_end)]
        to_upsert = list(zip(ids, emb, meta))
        _ = index.upsert(vectors=to_upsert)
