from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


tfid = TfidfVectorizer()

df = df.fillna('')

tfi_trns = tfid.fit_transform(df['nounize'])
tfi_trns = tfi_trns.astype('float32')
cos_smlt = cosine_similarity(tfi_trns, tfi_trns)#duplicate doc infection
df['similarity'] = ''

for idx, itm in enumerate(tqdm(df['nounize'], 'scoring similarity: ')):
  scr_smlt = list(enumerate(cos_smlt[idx]))
  scr_smlt = sorted(scr_smlt, key=lambda x: x[1], reverse=True)
  scr_smlt = [{i[0] : i[1]} for i in scr_smlt if (i[1] > 0.3) and (i[1]<0.9)]
  df['sim'].iloc[idx] = scr_smlt
