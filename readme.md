# Evaluation for GermEval2017 and MobASA

This evaluation script accepts GermEval2017, MobASA or XML file of the same format. The task to be evaluated are relevance, sentiment and opinion target extraction(OTE). Relevance and sentiment are text-level classification. We use classification report of sklearn. OTE belongs to token classification and is evaluated by seqeval.

To execute, run the following command:

```
python xmlEvaluation.py path/to/label path/to/prediction [relevance/sentiment/OTE]
```

For more details on the corpus, please refer to:

http://ltdata1.informatik.uni-hamburg.de/germeval2017/

https://github.com/DFKI-NLP/sim3s-corpus

The germeval_test_syn.xml is a modified version by the script here.

https://github.com/KaoCircle/modified_GermEval17
