import os
import pandas as pd
from argparse import ArgumentParser
import sklearn.metrics as slm
import seqeval.metrics as sem
from utils import xmlReader, string_offsets, find_iob


def taska(label, pred):
    print('Task to evaluate: text relevance\n')
    print(slm.classification_report(label.relevance.to_numpy(), pred.relevance.to_numpy(), target_names=['Irrelevant', 'Relevant'], digits=4))
    print('F1 score:', slm.f1_score(label.relevance.to_numpy(), pred.relevance.to_numpy()))



def taskb(label, pred):
    print('Task to evaluate: text sentiment\n')
    print(slm.classification_report(label.sentiment.to_numpy(), pred.sentiment.to_numpy(), target_names=['Negative', 'Neutral', 'Positive'], digits=4))


def taskd(label, pred):
    print('Task to evaluate: OTE(opinion target extraction)\n')
    label_iob = []
    pred_iob = []
    for i in range(len(label)):
        offsets = string_offsets(label.text[i])
        l_iob = find_iob(label.opinions[i], offsets)
        label_iob.append(l_iob)
        p_iob = find_iob(pred.opinions[i], offsets)
        pred_iob.append(p_iob)

    # we use default mode of seqeval, compatible with conlleval.
    # Please check github description for different settings
    # You also need to change the tagging format in function find_iob
    print(sem.classification_report(label_iob, pred_iob, digits=4))
    print('Accuracy score:', sem.accuracy_score(label_iob, pred_iob))


def main():
    """
    accept two XML file as true label and prediction
    evaluate the appointed subtask: relevance, sentiment or OTE
    relevance: binary text classification
    sentiment: multiclass text classification (negative, neutral, positive)
    OTE: short for opinion target extraction. token classification
    """

    parser = ArgumentParser(description="an evaluation script for GermEval 2017 and MobASA")
    parser.add_argument('truefile', help="True label filepath")
    parser.add_argument('predfile', help="Prediction filepath")
    parser.add_argument('task', help="Task to be evaluated, options=relevance, sentiment, OTE")

    args = parser.parse_args()

    if os.path.exists(args.predfile) and os.path.exists(args.truefile):
        print('Label file: '+args.truefile)
        print('Prediction file: '+args.predfile)

        # read xml file into dataframe
        truedf = xmlReader(args.truefile)
        preddf = xmlReader(args.predfile)

        # call respective task function
        if args.task.casefold() == 'relevance': taska(truedf, preddf)
        elif args.task.casefold() == 'sentiment': taskb(truedf, preddf)
        elif args.task.casefold() == 'ote': taskd(truedf, preddf)
        else:print('Task not recognized. Options=relevance, sentiment or OTE')
    else:
        print('File not exist!')


if __name__ == "__main__":
    main()
