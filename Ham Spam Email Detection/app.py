import  streamlit as st
import joblib
import  re
import string
import  nltk
from nltk.corpus import  stopwords
from  nltk.stem.porter import  PorterStemmer

port_stem=PorterStemmer()
def transform_text(text):
    transform_text=re.sub('[^a-zA-Z]',' ',text)
    transform_text=transform_text.lower()
    transform_text=nltk.word_tokenize(transform_text)  # split
    transform_text=[port_stem.stem(word) for word in transform_text if not word in stopwords.words('english')]
    transform_text=' '.join(transform_text)
    return transform_text


tfidf = joblib.load(open('vectorizer.joblib','rb'))
model = joblib.load(open('model.joblib','rb'))

st.title("Email/SMS spam classifier")

input_sms = st.text_input("Enter the message")

if st.button("predict"):

# cleaning
    trasform_sms=transform_text(input_sms)
# vectorize/preprocessing
    vector_input=tfidf.transform([trasform_sms])
# predic
    result=model.predict(vector_input)
# display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
