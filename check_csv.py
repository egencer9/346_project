import pandas as pd
df = pd.read_csv("data/named-entity-recognition-ner-corpus/ner.csv", encoding="latin1", nrows=5)
print(df.columns)
