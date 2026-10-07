import streamlit as st
import joblib

vectorizer = joblib.load('for transform.pkl')      
label = joblib.load('topredict.pkl')               
model = joblib.load('fack_new_dectorer.pkl')      


st.sidebar.title("ℹ️ About")
st.sidebar.info("""
Fake News Detector built using:
- **Python, Scikit-learn, TF-IDF**
- **Gradient Boosting Classifier**
- **Dataset: Fake vs Real news**

Developer: **Induri Avinash Reddy**
🔗 [GitHub](https://github.com/avinashreddy0)
🌐 [LinkedIn](https://www.linkedin.com/in/avinash-reddy-induri-4662b832a/)
""")

st.title('🚨 TruthSeeker AI – Fake News Detection System')
st.markdown("""
This app classifies whether a news article is **Real**  or **Fake** .
Paste your text below and click Predict.
""")


title = st.text_input('📝 Enter News Title', 'AI is the future of technology')
description = st.text_area('Enter News Description', 'AI research is rapidly growing in 2025.')

user_input = title + " " + description


if st.button("Predict"):
    # transform text
    input_data = vectorizer.transform([user_input])
    
    # predict
    prediction = model.predict(input_data)[0]
    prediction_label = label.inverse_transform([prediction])[0]
    
    # probability
    prob = model.predict_proba(input_data)[0]
    confidence = round(max(prob) * 100, 2)
    
    # display
    if prediction_label == "Fake":
        st.error(f" Prediction: Fake News ({confidence}% confidence)")
        st.markdown(f"<h3 style='color:red;'> Fake News ({confidence}%)</h3>", unsafe_allow_html=True)
    else:
        st.success(f" Prediction: Real News ({confidence}% confidence)")
        st.markdown(f"<h3 style='color:green;'> Real News ({confidence}%)</h3>", unsafe_allow_html=True)
