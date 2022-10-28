# Randomly splits identities to train eval split (make sure the bigrams are in the eval split)

# generate civil_test_data, civi_toxic_train_data, civil_train_data, synthetic_nontoxic_df, synthetic_toxic_df (and places them in the correct directory)

# also save the train identity terms in a csv file
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random


def create_splits(path):
    identities = pd.read_csv(path)

    identity_array = identities.to_numpy()
    identity_array = np.append(identity_array, 'lesbian')
    identity_list = identity_array.tolist()

    identity_list.remove('african american')
    identity_list.remove('middle aged')
    identity_list.remove('middle eastern')
    eval_ids = []
    eval_ids.append('african american')
    eval_ids.append('middle aged')
    eval_ids.append('middle eastern')

    for i in range(12):
        rand_num = np.random.randint(len(identity_list))
        ident = identity_list.pop(rand_num)
        eval_ids.append(ident)
        
    return identity_list, eval_ids
    

def generate_civil_data(toxic, path, identity_list):
    df = pd.read_csv(path)

    sentences = []
    a = []
    toxicity = []

    for row_index in tqdm(range(len(df))):
        comment_text = df.iloc[row_index]['comment_text'].split()
        if toxic:
            if df.iloc[row_index]['toxicity'] >= 0.5 and len(comment_text) <= 10 and len(set(identity_list).intersection(comment_text)) != 0:
                identity = str(set(identity_list).intersection(comment_text).pop())
                sentences.append(df.iloc[row_index]['comment_text'])
                toxicity.append(0)
                cur_a = []

                for diff_identity in identity_list:
                    cur_a.append(df.at[row_index, "comment_text"].replace(identity, diff_identity))
                a.append(cur_a)
        else:
            if df.iloc[row_index]['toxicity'] < 0.5 and len(comment_text) <= 10 and len(set(identity_list).intersection(comment_text)) != 0:
                identity = str(set(identity_list).intersection(comment_text).pop())
                sentences.append(df.iloc[row_index]['comment_text'])
                toxicity.append(0)
                cur_a = []

                for diff_identity in identity_list:
                    cur_a.append(df.at[row_index, "comment_text"].replace(identity, diff_identity))
                a.append(cur_a)

    return_df_raw = pd.DataFrame(list(zip(*a)))
    return_df = return_df_raw.T
    return_df.insert(0, column='comment_text', value=sentences)

    return return_df
    
def generate_synthetic_data(toxic, path, identity_list):
    df = pd.read_csv(path)

    sentences = []
    a = []
    toxicity = []

    for row_index in tqdm(range(len(df))):
        comment_text = df.iloc[row_index]['Text'].split()
        if toxic:
            if df.iloc[row_index]['Label'] == "BAD" and len(set(identity_list).intersection(comment_text)) != 0:
                identity = str(set(identity_list).intersection(comment_text).pop())
                sentences.append(df.iloc[row_index]['Text'])
                toxicity.append(0)
                cur_a = []

                for diff_identity in identity_list:
                    cur_a.append(df.at[row_index, "Text"].replace(identity, diff_identity))
                a.append(cur_a)
        else:
            if df.iloc[row_index]['Label'] == "NOT_BAD" and len(set(identity_list).intersection(comment_text)) != 0:
                identity = str(set(identity_list).intersection(comment_text).pop())
                sentences.append(df.iloc[row_index]['Text'])
                toxicity.append(0)
                cur_a = []

                for diff_identity in identity_list:
                    cur_a.append(df.at[row_index, "Text"].replace(identity, diff_identity))
                a.append(cur_a)

    return_df_raw = pd.DataFrame(list(zip(*a)))
    return_df = return_df_raw.T
    return_df.insert(0, column='comment_text', value=sentences)

    return return_df

def main():
    train_ids, test_ids = create_splits("../data/adjectives_people.txt")

    civil_test = generate_civil_data(False, "../data/civil_comments/civil_comments.csv", test_ids)
    civil_train_toxic = generate_civil_data(True, "../data/civil_comments/civil_comments.csv", train_ids)
    civil_train_nontoxic = generate_civil_data(False, "../data/civil_comments/civil_comments.csv", train_ids)
    
    synthetic_toxic = generate_synthetic_data(True, "../data/bias_madlibs_89k.csv", train_ids)
    synthetic_nontoxic = generate_synthetic_data(False, "../data/bias_madlibs_89k.csv", train_ids)
    
    with open("../data/random_split_data/train_identities.txt", "w") as output:
        for i in train_ids:
            output.write(str(i) + "\n")

    with open("../data/random_split_data/test_identities.txt", "w") as output:
        for i in test_ids:
            output.write(str(i) + "\n")
    
    civil_test.to_csv(Path("../data/random_split_data/civil_test_data.csv"))
    civil_train_toxic.to_csv(Path("../data/random_split_data/civil_toxic_train_data.csv"))
    civil_train_nontoxic.to_csv(Path("../data/random_split_data/civil_train_data.csv"))
    
    synthetic_toxic.to_csv(Path("../data/random_split_data/synthetic_toxic_df.csv"))
    synthetic_nontoxic.to_csv(Path("../data/random_split_data/synthetic_nontoxic_df.csv"))
    
main()