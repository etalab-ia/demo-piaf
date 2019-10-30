# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import time, re
import wikipedia

from multiprocessing import freeze_support

from cdqa.utils.download import download_model
from cdqa.pipeline.cdqa_sklearn import QAPipeline

if __name__ == '__main__':
	freeze_support()

	st.title('Un démonstrateur pour PIAF')
	st.write("Ceci est un démonstrateur du potentiel d'un système de Question Answering, ici en anglais.\
			  Posez vos questions sur un paragraphe de texte et admirez le résultat ! \N{bird}")
	
	langu = st.sidebar.selectbox("Langue", ["Anglais", "Français (en développement)"])
	# mod = st.sidebar.selectbox("Modèle", ["Bert SQuAD 1.1", "Bert SQuAD 2.0"])
	source = st.sidebar.selectbox("Source", ["Un article Wikipédia au choix", "Un paragraphe de votre cru"])
	
	if "Wikipédia" in source:
	
		pagename = st.sidebar.text_area("Nom de l'article", 'Elizabeth II')
		
		title, id, paragraphs = [], [], []
		page = wikipedia.page(pagename)
		title.append(page.title)
		id.append(page.pageid)
		content = re.split('\n{1,3}={2,3}\s.+\s={2,3}\n{1,2}|\n', page.content)
		paragraphs.append([s for s in content if s != ''])
		
		df = pd.DataFrame({'id': id, 'title': title, 'paragraphs': paragraphs})
		default_query = 'What is {}?'.format(pagename)
		
		st.header('Posez vos questions sur l\'article : {}'.format(page.title))
		st.write("Introduction : *{}*".format(df.loc[0,'paragraphs'][0]))
		
	else:
	
		default = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. \
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
		para = st.text_area('Ecrivez ici le paragraphe source', default)
		df = pd.DataFrame([[0, 'My paragraph', [para]]], columns=['id', 'title', 'paragraphs'])
		
		default_query = 'What is it?'
	
	### MODEL TRAINING SECTION ###
	
	# Download data and models
	download_model(model='bert-squad_1.1', dir='./models')
	
	s1 = time.time()
	# Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1
	cdqa_pipeline = QAPipeline(reader='models/bert_qa_vCPU-sklearn.joblib', max_df=1.0, min_df=1)
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
	st.write('Paragraphe de l\'article : *{}*\n'.format(prediction[2]))
	st.write('Répondre à votre question a nécessité ', round(t3), ' secondes, charger le modèle', round(t1), 'secondes.')
	