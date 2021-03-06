# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import time, re
import wikipedia

from multiprocessing import freeze_support

from cdqa.utils.download import download_model
from cdqa.pipeline.cdqa_sklearn import QAPipeline

HTML_WRAPPER = """<span style = "overflow-x: auto;
                                 color: white;
                                 background-color: rgb(246, 51, 102);
                                 border: 1px solid #e6e9ef;
                                 border-radius: 0.4rem;
                                 padding: 0.2rem;
                                 margin-bottom: 2.5rem">{}</span>"""

if __name__ == '__main__':
    freeze_support()

    st.title('Un démonstrateur pour PIAF')
    st.write("Ceci est un démonstrateur du potentiel d'un système de Question Answering, ici en anglais.\
              Posez vos questions sur un paragraphe de texte et admirez le résultat ! \N{bird}")
    
    langu = st.sidebar.selectbox("Langue", ["Français", "Anglais"])
    # mod = st.sidebar.selectbox("Modèle", ["Bert SQuAD 1.1", "Bert SQuAD 2.0"])
    source = st.sidebar.selectbox("Source", ["Le RGPD", "Un article Wikipédia au choix", "Un paragraphe de votre cru"])
    
    if "Wikipédia" in source:
    
        if "Français" in langu:
            wikipedia.set_lang("fr")
            pagename = st.sidebar.text_area("Nom de l'article", 'Élisabeth II')
        else:
            wikipedia.set_lang("en")
            pagename = st.sidebar.text_area("Nom de l'article", 'Elizabeth II')
        
        title, id, paragraphs = [], [], []
        page = wikipedia.page(pagename)
        title.append(page.title)
        id.append(page.pageid)
        content = re.split('\n{1,3}={2,3}\s.+\s={2,3}\n{1,2}|\n', page.content)
        paragraphs.append([s for s in content if s != ''])
        
        df = pd.DataFrame({'id': id, 'title': title, 'paragraphs': paragraphs})
        
        if "Français" in langu:
            default_query = "Qu'est ce que {} ?".format(page.title)
        else:
            default_query = 'What is {}?'.format(page.title)
        
        st.header('Posez vos questions sur l\'article : {}'.format(page.title))
        st.write("Introduction : *{}*".format(df.loc[0,'paragraphs'][0]))
    
    elif 'RGPD' in source:
        
        default_query = "Who can access my personal data?"
        pattern = re.compile('Article\s\d{1,2}\.*')
        f = open("data/GDPR.txt", "r", encoding='utf-8')
        content = f.read().split('\n\n')
        content = [c for c in content if pattern.match(c)]
        df = pd.DataFrame([[0, 'RGPD', content]], columns=['id', 'title', 'paragraphs'])
    
    else:
    
        if "Français" in langu:
            default = "Le recours à l’intelligence artificielle au sein de l’action publique est souvent identifié comme une opportunité pour interroger des textes documentaires et réaliser des outils de questions/réponses automatiques à destination des usagers. Interroger le code du travail en langage naturel, mettre à disposition un agent conversationnel pour un service donné, développer des moteurs de recherche performants, améliorer la gestion des connaissances, autant d’activités qui nécessitent de disposer de corpus de données d’entraînement de qualité afin de développer des algorithmes de questions/réponses. Aujourd’hui, il n’existe pas de jeux de données d’entraînement francophones publics et ouverts qui permettrait d’entraîner ces algorithmes. L’ambition du projet PIAF est de construire ce(s) jeu(x) de données francophones pour l’IA de manière ouverte et contributive"
            default_query = 'Quel est le but de PIAF ?'
        else:
            default = "The use of artificial intelligence in public action is often identified as an opportunity to interrogate documentary texts and to create automatic question / answer tools for users. Querying natural language work code, providing a conversational agent for a given service, developing high-performance search engines, improving knowledge management, all activities that require quality training data corpus to develop question and answer algorithms. Today, there are no public and open French training data sets that would train these algorithms. The ambition of the PIAF project is to build this set of Francophone data for AI in an open and contributive way."
            default_query = 'What is the aim of PIAF?'
        
        para = st.text_area('Ecrivez ici le paragraphe source', default)
        df = pd.DataFrame([[0, 'My paragraph', [para]]], columns=['id', 'title', 'paragraphs'])
        
    
    ### MODEL TRAINING SECTION ###
        
    s1 = time.time()

    if not "Français" in langu:
        download_model(model='bert-squad_1.1', dir='./models')
        cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib', max_df=1.0, min_df=1)
    else:
        cdqa_pipeline = QAPipeline(reader='models/bert_qa_fr.joblib', max_df=1.0, min_df=1)
        
    # cdqa_pipeline.cuda()
    t1 = time.time() - s1

    s2 = time.time()
    # Fitting the retriever to the list of documents in the dataframe
    cdqa_pipeline.fit_retriever(df)
    t2 = time.time() - s2

    # Querying and displaying
    query = st.text_area('Posez votre question ici !', default_query)
    
    s3 = time.time()
    prediction = cdqa_pipeline.predict(query)
    t3 = time.time() - s3
    
    st.header('Réponse : {}\n'.format(prediction[0]))
    # st.write('Article d\'où la réponse est extraite : *{}*\n'.format(prediction[1]))
    res = prediction[2].replace(prediction[0], HTML_WRAPPER.format(prediction[0]))
    if "Wikipédia" in source:
        st.write('Paragraphe de l\'article : *{}*\n'.format(res), unsafe_allow_html=True)
    elif "RGPD" in source:
        st.write('Article concerné : *{}*\n'.format(res), unsafe_allow_html=True)
    else:
        st.write('Localisation de la réponse : *{}*\n'.format(res), unsafe_allow_html=True)
    
    st.write('Répondre à votre question a nécessité ', round(t3), ' secondes, charger le modèle', round(t1), 'secondes.')
    
