import numpy as np


#!pip install keybert

def BERT(lst):
    print(lst)
    # array_text = np.array(label)/
    array_text = pd.DataFrame(lst, columns=['nouns']).to_numpy()

    bow = []
    from keybert import KeyBERT
    kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
    for j in range(len(array_text)):
        keywords = kw_extractor.extract_keywords(array_text[j][0])
        bow.append(keywords)

    new_bow = []
    for i in range(0, len(bow)):
        for j in range(len(bow[i])):
            new_bow.append(bow[i][j])

    # print(new_bow)

    keyword = pd.DataFrame(new_bow, columns=['keyword', 'weight'])
    group = keyword.groupby(['keyword']).agg("sum").reset_index()

    sort = group.sort_values('weight', ascending=False)

    return sort['keyword'].head(100).to_list()


df_rslt = pd.DataFrame()

for res in res_kds:
    rslt = BERT(dic_res[res])
    df_rslt[res] = rslt

