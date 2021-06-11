# Ignore all your warnings
import os
import pickle
import warnings

import docx
import pandas as pd
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from tqdm import tqdm

warnings.filterwarnings("ignore")


def train_model(train_data, n_iter, drop_rate):
    """
    Fine-tune Spacy-NER model.
    Code Refer: https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
    """
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        index = 0
        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                try:
                    nlp.update([text],
                               [annotations],
                               drop=drop_rate,
                               sgd=optimizer,
                               losses=losses)
                except Exception as e:
                    pass
            print("Iteration " + str(itn + 1) + f" -- {str(losses)}")
    return None


def score(spacy_format_data, model):
    """ Function to clacluate recall metric of a model"""
    scorer = Scorer()
    try:
        for input_, annot in spacy_format_data:
            doc_gold_text = model.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annot['entities'])
            pred_value = model(input_)
            scorer.score(pred_value, gold)
    except Exception as e:
        print(e)
    return scorer.scores['ents_r']


# Merge Rental-Agreement doc files and entities
def extract_doc_text(doc_list):
    """ Given doc file names, return dataframe of extracted text"""
    file_names = []
    texts = []

    for file in tqdm(doc_list):
        file_name = file.rstrip('.pdf.docx').split('/')[-1]

        doc = docx.Document(file)
        full_text_list = [paragraph.text for paragraph in doc.paragraphs]
        full_text = " ".join(full_text_list)

        file_names.append(file_name)
        texts.append(full_text)

    return pd.DataFrame({'File Name': file_names, 'text': texts})


if __name__ == '__main__':
    MODEL_NAME = 'tmp_model'

    # Load TrainingTestSet and ValidationSet
    TrainingTestSet = pd.read_csv('../data/TrainingTestSet .csv')
    ValidationSet = pd.read_csv('../data/ValidationSet.csv')

    # Load Training rental agreements docx file
    Training_data_docx = os.listdir('../data/Training_data')
    Validation_Data_docx = os.listdir('../data/Validation_Data')

    training_data_docx_list = ["../data/Training_data/" + docx for docx in Training_data_docx]
    Validation_Data_docx_list = ["../data/Validation_Data/" + docx for docx in Validation_Data_docx]

    # Extracting text from doc file
    train_text = extract_doc_text(training_data_docx_list)
    val_text = extract_doc_text(Validation_Data_docx_list)

    # Merge Rental-Agreement doc files and entities from csv
    train_data = pd.merge(TrainingTestSet, train_text, on='File Name', how='inner')
    val_data = pd.merge(ValidationSet, val_text, on='File Name', how='inner')

    # loading training annotated data
    with open('../data/mannual_annotated/train_annotations.pkl', 'rb') as f:
        mannual_annotated_train_df = pickle.load(f)

    # loading validation annotated data
    with open('../data/mannual_annotated/val_annotations.pkl', 'rb') as f:
        mannual_annotated_val_df = pickle.load(f)

    train_data['mannal_annotation'] = mannual_annotated_train_df
    val_data['mannal_annotation'] = mannual_annotated_val_df

    mannual_annotated_train_df_without_null = [x for x in train_data['mannal_annotation'] if len(x) > 0]
    mannual_annotated_val_df_without_null = [x for x in val_data['mannal_annotation'] if len(x) > 0]

    import random

    # Number of iteration to train the model
    n_iter = 100
    drop_rate = 0.4  # Drop Rate
    nlp = spacy.blank('en')

    # Training spacy model on pseudo annotated data
    print('Training Start ..')
    train_model(mannual_annotated_train_df_without_null, n_iter, drop_rate)

    # Saving the model
    nlp.to_disk(MODEL_NAME)

    # loading the model
    nlp_model3 = spacy.load(MODEL_NAME)

    # Recall on training data - Pseudo annotated data
    training_recall_model3 = score(mannual_annotated_train_df_without_null, nlp_model3)
    print(f"Recall on validation data: {training_recall_model3}")

    # Recall on val data - Pseudo annotated data
    val_recall_model3 = score(mannual_annotated_val_df_without_null, nlp_model3)
    print(f"Recall on validation data: {val_recall_model3}")
