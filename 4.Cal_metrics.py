from nltk.tokenize import word_tokenize
from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score
import os
import csv
from rouge import Rouge
import numpy as np


def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score


def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores


def read_file(file_name):
    f = open(file_name, "r", encoding="utf-8")

    refs = []
    cands = []
    reader = csv.reader(f)

    for i, line in enumerate(reader):
        # if len(line[0].split("Assistant:")) != 1:
        #     cands.append(line[0].split("Assistant:")[1])
        # else:
        #     cands.append([])
        #     print(111)

        refs.append(line[1])

    return refs, cands


if __name__ == "__main__":
    files = [
        'results/results.csv'
    ]

    with open("results/Report_Greedy1.csv", "w", newline='') as reportFile:
        fp_write = csv.writer(reportFile)
        fp_write.writerow(
            ['Model', 'Distinct-1', 'Distinct-2', 'Bleu', 'Rouge-l', 'F-Bert', 'P-Bert', 'R-Bert', 'CHRF'])
        for f in files:
            print(f"Evaluating {f}")
            refs, cands = read_file(f)

            dist_1, dist_2 = calc_distinct(cands)

            bleu = BLEU().corpus_score(cands, [refs])
            print(f"***** BLEU: {bleu.score} *****")

            for num, can in enumerate(cands):
                if len(can) == 0:
                    print(num)
                    cands[num] = 'a'

            scores = Rouge().get_scores(cands, refs)
            print(f"***** Rouge-f : {np.mean([score['rouge-l']['f'] for score in scores]) * 100} *****")

            P_Bert, R_Bert, F_Bert = score(cands, refs, lang="en", rescale_with_baseline=True, model_type='D:\Project\LLM\Roberta-large')
            print(f"***** Bert_P: {P_Bert.mean() * 100} *****")
            print(f"***** Bert_R: {R_Bert.mean() * 100} *****")
            print(f"***** Bert_F: {F_Bert.mean() * 100} *****")

            chrf = CHRF().corpus_score(cands, [refs])
            print(f"***** CHRF: {chrf.score} *****")

            result = ['#' + str(f),
                      round(dist_1 * 100, 4), round(dist_2 * 100, 4),
                      round(bleu.score, 4),
                      round(np.mean([score['rouge-l']['f'] for score in scores]) * 100, 4),
                      round(float(F_Bert.mean()) * 100, 4), round(float(P_Bert.mean())* 100, 4), round(float(R_Bert.mean()) * 100, 4),
                      round(chrf.score, 4)]
            fp_write.writerow(result)
