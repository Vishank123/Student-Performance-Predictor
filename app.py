import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("student_data.csv")

studytime = st.slider("Study Time",0,10)
absences = st.slider("Absences",0,50)
G1 = st.slider("First Test Marks",0,20)
G2 = st.slider("Second Test Marks",0,20)

X = data[['studytime','absences','G1','G2']]
y = data['G3']

# Train model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = r2_score(y_test,pred)

# Streamlit UI
st.title("Student Performance Predictor")

if st.button("Predict Score"):
    prediction = model.predict([[studytime, absences, G1, G2]])
    st.write("Predicted Final Score:", prediction[0])

st.write("Model Accuracy:", accuracy)


#streamlit run app.py run this command on terminal and enjoy the model
