import pandas as pd
from src.database_mysql import DatabaseManager
from src.tfidf_vectorizer import TFIDFFeatureExtractor


db = DatabaseManager()

query = """
SELECT
url,
text_content,
url_length,
num_dots,
num_hyphen,
num_slashes,
https,
subdomains,
has_at_symbol,
has_ip,
has_ssl,
ssl_valid,
text_length,
token_count,
scam_keyword_count,
scam_keyword_density
FROM websites
"""

db.cursor.execute(query)

rows = db.cursor.fetchall()

data = []

texts = []

for row in rows:

    text = row[1] if row[1] else ""

    texts.append(text)

    data.append({
        "url": row[0],
        "url_length": row[2],
        "num_dots": row[3],
        "num_hyphen": row[4],
        "num_slashes": row[5],
        "https": row[6],
        "subdomains": row[7],
        "has_at_symbol": row[8],
        "has_ip": row[9],
        "has_ssl": row[10],
        "ssl_valid": row[11],
        "text_length": row[12],
        "token_count": row[13],
        "scam_keyword_count": row[14],
        "scam_keyword_density": row[15]
    })


# convert to dataframe
df_features = pd.DataFrame(data)


# TF-IDF
tfidf = TFIDFFeatureExtractor()

tfidf_matrix = tfidf.fit_transform(texts)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names()
)

# combine all features
final_dataset = pd.concat([df_features, tfidf_df], axis=1)


# save dataset
final_dataset.to_csv("ml_project_dataset.csv", index=False)

print("ML dataset created: ml__project_dataset.csv")