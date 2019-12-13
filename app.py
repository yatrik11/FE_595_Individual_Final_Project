#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import io
import urllib, base64
from io import BytesIO
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("FE595_Final_Project/spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)

	#transforming text to an array
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

@app.route('/explore')

def explore():

	df= pd.read_csv("/FE595_Final_Project/spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	count=df['class'].value_counts()
	plt.bar(['spam','ham'],count)
	plt.legend(labels=['spam','ham'])
	plt.ylabel('No of Class', fontsize=10)
	plt.title('Spam vs Ham')
	plt.tight_layout()
	plt.savefig('FE595_Final_Project/plot.jpg', dpi=300, bbox_inches='tight')

	#Rendering Plot in Html
	figfile = BytesIO()
	plt.savefig(figfile, format='png')
	figfile.seek(0)
	figdata_png = base64.b64encode(figfile.getvalue())
	result = str(figdata_png)[2:-1]
	return render_template('result.html', result=result)
 
if __name__ == '__main__':
	app.run(debug=True)


# In[ ]:




