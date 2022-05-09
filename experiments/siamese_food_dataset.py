import pandas as pd
from itertools import combinations


def generate_pairs():
    data = pd.read_parquet("../data/menu000000000000.parquet")
    comb = data.groupby("id_section")["item_id"].apply(lambda x: list(combinations(x, 2)))
    positive = pd.DataFrame(comb.explode("item_id").dropna().to_list())
    positive.columns = ["item_a", "item_b"]
    positive["target"] = 0
    negative = positive[["item_a"]].copy()
    negative["item_b"] = data["item_id"].sample(negative.shape[0], replace=True).reset_index(drop=True)
    negative["target"] = 1

    pairs = pd.concat([positive, negative], axis=0).reset_index(drop=True)
    pairs = pairs.sample(frac=1).reset_index(drop=True)
    pairs.to_csv("../data/pairs.csv", index=False)


generate_pairs()
base = pd.read_parquet("../data/menu000000000000.parquet")

base["section_name"] = base["section_name"].fillna("").str.replace("\n", " ")
base["product_name"] = base["product_name"].fillna("").str.replace("\n", " ")
base["product_description"] = base["product_description"].fillna("").str.replace("\n", " ")

(
        base["section_name"].fillna("")
        + " "
        + base["product_name"].fillna("")
        + " "
        + base["product_description"].fillna("")
).str.replace("\n", " ").to_csv('../data/product.csv', index=False, sep=";", header=False)

pairs_raw = pd.read_csv("../data/pairs.csv", chunksize=100_000)
pairs = next(pairs_raw)
for idx, pairs in enumerate(pairs_raw):
    print(idx)
    columns = ["item_id", "product_name", "product_description", "section_name"]
    pairs = pairs.merge(base[columns], left_on="item_a", right_on="item_id", how="left")
    pairs = pairs.drop(columns=["item_id"])

    pairs = pairs.merge(base[columns], left_on="item_b", right_on="item_id", how="left", suffixes=("_a", "_b"))
    pairs = pairs.drop(columns=["item_id"])

    out_columns = [
        "item_a",
        "product_name_a",
        "product_description_a",
        "section_name_a",
        "item_b",
        "product_name_b",
        "product_description_b",
        "section_name_b",
        "target"
    ]
    pairs[out_columns].to_csv(f'../data/pairs/dataset-{idx:05}.csv', index=False, sep=";")
