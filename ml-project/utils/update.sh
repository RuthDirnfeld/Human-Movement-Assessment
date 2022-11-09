#!/bin/bash
URL_AS=https://gist.githubusercontent.com/MatusMaruna/a404fa34063e7d95d8ce5af549992a2d/raw/6ed472b099522e70530da326b6a7045e2af8a638/AimoScore_WeakLink_big_scores.csv
URL_WS=https://gist.githubusercontent.com/MatusMaruna/1e94a8718e23d596a2ba474f9775fb80/raw/d3718bb7fba77cd2bfab185389dc7e243efa2f02/20190108%2520scores_and_weak_links.csv
URL_EXPERT_SCORE_TRAIN=https://gist.githubusercontent.com/MatusMaruna/563e97ceab42ca37ce44e1192bd5e9e6/raw/0d82f2fa7a2e24e700a49fae9abcfcceb49529fb/expert_score_train_set.csv
URL_EXPERT_SCORE_VALID=https://gist.githubusercontent.com/MatusMaruna/d92ef0270dc786804e3d71ca4b0591dd/raw/4e8926355632da997120c29f31f381f418172d3c/expert_score_validation_set.csv
DIR=../data/datasets/
if [ ! -d $DIR ]; then
  mkdir -p $DIR;
fi
pushd $DIR
curl -o  AimoScore_WeakLink_big_scores.csv $URL_AS
curl -o 20190108scores_and_weak_links.csv $URL_WS
curl -o  expert_score_train_set.csv $URL_EXPERT_SCORE_TRAIN
curl -o expert_score_validation_set.csv $URL_EXPERT_SCORE_VALID
popd
python3 preprocessing.py $DIR