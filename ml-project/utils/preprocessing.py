import argparse

import pandas as pd

parser = argparse.ArgumentParser(description='Pre-processing for datasets')
parser.add_argument('dirName', metavar='N1', type=str,
                    help='Directory for reading/saving datasets')

args = parser.parse_args()

data = pd.read_csv(args.dirName + 'AimoScore_WeakLink_big_scores.csv', decimal=',')
weaklinks = pd.read_csv(args.dirName + '20190108scores_and_weak_links.csv', decimal=',')
idx = weaklinks.drop(['Unnamed: 0', 'Date', 'SCORE', 'ID'], axis=1).astype(float).idxmax(axis=1)
max = weaklinks.drop(['Unnamed: 0', 'Date', 'SCORE', 'ID'], axis=1).astype(float).max(axis=1)

d = pd.DataFrame(0, index=weaklinks['ID'], columns=['WeakLink_label', 'WeakLink_score'])

for i in range(len(idx.array)):
    d.at[weaklinks.iloc[i]['ID']] = str(idx.array[i])
    d.at[weaklinks.iloc[i]['ID'], 'WeakLink_score'] = max.array[i]

new_dataset_labels_score = pd.merge(data, d, left_on='ID', right_on='ID', how='left')
new_dataset_labels_score.to_csv(args.dirName + 'AimoScore_WeakLink_big_scores_Labels_and_Scores.csv', decimal=',')

d = pd.DataFrame(0, index=weaklinks['ID'], columns=['WeakLink_score'])

for i in range(len(idx.array)):
    d.at[weaklinks.iloc[i]['ID']] = max.array[i]

new_dataset_score = pd.merge(data, d, left_on='ID', right_on='ID', how='left')
new_dataset_score.to_csv(args.dirName + 'AimoScore_WeakLink_big_scores_Scores.csv', decimal=',')

d = pd.DataFrame(0, index=weaklinks['ID'], columns=['WeakLink_label'])

for i in range(len(idx.array)):
    d.at[weaklinks.iloc[i]['ID']] = str(idx.array[i])

new_dataset_labels = pd.merge(data, d, left_on='ID', right_on='ID', how='left')
new_dataset_labels.to_csv(args.dirName + 'AimoScore_WeakLink_big_scores_labels.csv', decimal=',')

d = pd.DataFrame(0, index=weaklinks['ID'], columns=range(weaklinks.shape[1]))
weaklinksM = weaklinks.drop(['Unnamed: 0', 'Date', 'SCORE', 'ID'], axis=1)
for i in range(len(idx.array)):
    d.iat[i, weaklinksM.columns.get_loc(idx.array[i])] = 1

d['One_hot_encoding'] = d[d.columns[0:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
d = d.drop(columns=d.columns[0:18])

new_dataset_one_hot = pd.merge(data, d, left_on='ID', right_on='ID', how='left')
new_dataset_one_hot.to_csv(args.dirName + 'AimoScore_WeakLink_big_scores_one_hot.csv', decimal=',')
