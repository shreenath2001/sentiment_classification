import streamlit as st
import pandas as pd
import pickle
import nb_classifier
import mlp_classifier

def main():

	f = open('NB_classifier.pickle', 'rb')
	classifier_nb = pickle.load(f)
	f.close()

	f = open('MLP_classifier.pickle', 'rb')
	classifier_mlp = pickle.load(f)
	f.close()

	@st.cache(persist = True)
	def load_data():
		data = pd.read_csv("train.txt")
		data.drop('tweet_id', inplace = True, axis = 1)
		data.set_index('tweet_text', drop = True)
		return data

	data = load_data()


	st.title("TEAM MATRIX")
	html_1 = """
				<div style = "background:Periwinkle">
	"""
	st.markdown(html_1, unsafe_allow_html = True)
	html_2 = """
				<div style = "background-color:yellow;padding:10px">
				<h2 style = "color:black;text-align:center;">Sentiment Classification App</h2>
				</div>
	"""
	st.markdown(html_2, unsafe_allow_html = True)
	st.markdown("Classifies the sentence to be positive, negative or neutral")

	classifier = st.selectbox("select classifier?", ("Multi layer perceptron", "Naive Bayes"))

	if classifier == "Multi layer perceptron":

		user_input = st.text_input("Enter your text")
		if st.button("Analyze"):
			input_val = mlp_classifier.vect([user_input])
			ans = classifier_mlp.predict(input_val)
			if(ans == 1):
				val = "positive"
			elif(ans == -1):
				val = "negative"
			else:
				val = "neutral"
			st.success("Given sentence is: {}".format(val))
			

	if classifier == "Naive Bayes":

		user_input = st.text_input("Enter your text")
		if st.button("Analyze"):
			input_val = nb_classifier.vect([user_input])
			ans = classifier_nb.predict(input_val)
			if(ans == 1):
				val = "positive"
			elif(ans == -1):
				val = "negative"
			else:
				val = "neutral"
			st.success("Given sentence is: {}".format(val))

	if st.checkbox("Show training data", False):
		st.subheader("This training data is from kaggle competition")
		st.write(data)

	st.sidebar.markdown("This application is created using the data from the kaggle competition Sentiment Analysis of tweet")
	st.sidebar.markdown("Go To [Competition](https://www.kaggle.com/c/sentiment-analysis-of-tweets)")



if __name__ == '__main__':
	main()