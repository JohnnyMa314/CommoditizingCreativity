import pandas as pd
import spacy
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from FeatureUtils import tokenize_script
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

def main():
    script_info = pd.read_csv('./data/IMSDB/final_movie_budgets.csv', sep=',')
    script_info['Budget'] = [int(bud.replace(',', '')) for bud in script_info['Budget']]  # reformatting budget

    # creating Budget Categories by quartile
    script_info['Bud_Cat'] = pd.qcut(script_info['Budget'], 2, labels=[0, 1])

    # get list of scripts from data folder
    scripts = []
    for file in script_info['Filename']:
        with open(file, 'r') as txt:
            scripts.append(txt.read().replace('\n', ''))

    X_train, X_test, y_train, y_test = train_test_split(scripts, script_info['Bud_Cat'], test_size=0.2, random_state=0)

    docs = [' '.join(tokenize_script(script, stop_words=True)) for script in X_train]
    train_docs = [list(x) for x in zip(docs, y_train)]

    train_df = pd.DataFrame(train_docs)
    train_df.columns = ["text", "labels"]

    docs = [' '.join(tokenize_script(script, stop_words=True)) for script in X_test]
    test_docs = [list(x) for x in zip(docs, y_test)]

    test_df = pd.DataFrame(test_docs)
    test_df.columns = ["text", 'labels']

    model_args = ClassificationArgs(sliding_window=True, overwrite_output_dir=True)

    model = ClassificationModel(
        "roberta",
        "roberta-base",
        args=model_args, use_cuda=True, n_epochs = 3)

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_df)

    print(result)

if __name__ == '__main__':
    main()
