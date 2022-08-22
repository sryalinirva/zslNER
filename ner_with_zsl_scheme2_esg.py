#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import numpy as np
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
nlp=spacy.load('en_core_web_sm')

classifier = pipeline("zero-shot-classification")

def get_zsl_entities_scheme2_esg(text,ESG_terms,sub_labels,domain_label):

    results = classifier(text, ESG_terms, multi_class=True)
    scores = np.asarray(results['scores'])
    labels = np.asarray(results['labels'])
    idx = scores >= 0.7
    KPI_label = list(labels[idx])
    # Retokenize sentence with noun phrases as one token
    text_doc = nlp(text)
    # noun_phrases = list(text_doc.ents)
    noun_phrases = list(text_doc
                        .ents) + list(text_doc.noun_chunks)
    noun_phrases = spacy.util.filter_spans(noun_phrases)
    with text_doc.retokenize() as retokenizer:
        [retokenizer.merge(span) for span in noun_phrases]

    # Get the NER Labels for the noun phrases
    entities_labels = {}
    for ent in text_doc.ents:
        entities_labels[ent.text] = ent.label_
    #Populate Labels for the sentence
    sent_tokens = []
    zsl_labels = []
    spacy_ner_labels = []
    for token in text_doc:
        sent_tokens.append(token.text)
        zsl_labels.append('O')
        if token.text in list(entities_labels):
            spacy_ner_labels.append(entities_labels[token.text])
        else:
            spacy_ner_labels.append('O')

    expanded_labels = KPI_label + sub_labels
    #Assign the Label KPI/Unit for the test phrases if they are indicative of the KPI/Unit
    potential_phrases = []
    scores = []
    phrase_labels = []
    for test_phrase in noun_phrases:
        results = classifier(str(test_phrase), expanded_labels, multi_class=True)
        temp_scores = np.asarray(results['scores'])
        temp_ix = temp_scores >= 0.6
        temp_scores = temp_scores[temp_ix]
        if len(temp_scores) > 0:
            potential_phrases.append(str(test_phrase))
            scores.append(temp_scores[0])
            phrase_labels.append(domain_label)

    for i,token in enumerate(sent_tokens):
        if str(token) in potential_phrases:
            zsl_labels[i] = domain_label
    #     print(token,'--->',zsl_labels[i],spacy_ner_labels[i])
    words = []
    word_level_zsl_ner_labels = []
    word_level_spacy_ner_labels = []

    for i,token in enumerate(sent_tokens):
        temp_doc = nlp(token)
        token_splits = [token.text for token in temp_doc ]
        zsl_label = zsl_labels[i]
        spacy_ner_label = spacy_ner_labels[i]
        words.extend(token_splits)
        word_level_zsl_ner_labels.extend([zsl_label]*len(token_splits))
        word_level_spacy_ner_labels.extend([spacy_ner_label]*len(token_splits))

    #Cleanup labels using POS Tags
    text_doc = nlp(text)
    for i, token in enumerate(text_doc):
        if token.is_punct or token.is_stop:
            word_level_zsl_ner_labels[i] = 'O'
        # if word_level_zsl_ner_labels[i] == domain_label:
        #     if token.pos_ not in ['PROPN','NOUN']:
        #         word_level_zsl_ner_labels[i] = 'O'
        if word_level_zsl_ner_labels[i] == 'O':
            word_level_zsl_ner_labels[i] = word_level_spacy_ner_labels[i]
    #     print(i,token,'--->',word_level_zsl_ner_labels[i])

    word_level_zsl_ner_labels = np.asarray(word_level_zsl_ner_labels)
    ix_label = np.where(word_level_zsl_ner_labels == domain_label)[0]
    for i, token in enumerate(text_doc):
        if i - 1 in ix_label:
            if not token.is_alpha and not (token.is_punct or token.is_stop):
                word_level_zsl_ner_labels[i] = domain_label
            if not (token.is_punct or token.is_stop):
                word_level_zsl_ner_labels[i] = domain_label
    #     print(i,token,'--->',word_level_zsl_ner_labels[i])

    #BIULO Scheme
    zsl_ner_labels = ['O']*len(word_level_zsl_ner_labels)
    for i, label in enumerate(word_level_zsl_ner_labels):
        if i in range(1,len(word_level_zsl_ner_labels)-1):
            if label != 'O':
                  #Intermediate token in the entity
                if word_level_zsl_ner_labels[i-1] == label and word_level_zsl_ner_labels[i+1] == label:
                    zsl_ner_labels[i] = 'I-' + label
                elif word_level_zsl_ner_labels[i-1] == label and word_level_zsl_ner_labels[i+1] != label:
                    #Last token in the entity
                    zsl_ner_labels[i] = 'L-' + label
                elif word_level_zsl_ner_labels[i-1] != label and word_level_zsl_ner_labels[i+1] == label:
                    #Begining of the entity
                    zsl_ner_labels[i] = 'B-' + label
                elif word_level_zsl_ner_labels[i-1] != label and word_level_zsl_ner_labels[i+1] != label:
                    # Single token entity
                    zsl_ner_labels[i] = 'U-' + label
        # First Token
        elif i == 0 and label != 'O':
            if word_level_zsl_ner_labels[i+1] != label:
                # Single token entity
                zsl_ner_labels[i] = 'U-' + label
            elif word_level_zsl_ner_labels[i+1] == label:
                #Begining of the entity
                zsl_ner_labels[i] = 'B-' + label
        # Last Token
        elif i == len(word_level_zsl_ner_labels)-1 and label != 'O':
            if word_level_zsl_ner_labels[i-1] != label:
                # Single token entity
                zsl_ner_labels[i] = 'U-' + label
            elif word_level_zsl_ner_labels[i-1] == label:
                # Last token of the entity
                zsl_ner_labels[i] = 'L-' + label

        # print(i,words[i],'--->',zsl_ner_labels[i],word_level_zsl_ner_labels[i])

    return words, zsl_ner_labels
# In[ ]:




