import streamlit as st
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
#PAGE CONFIG
st.set_page_config("Linear Regression",layout="centered")

#Load CSS


def load_css(file):
    if os.path.exists(file):
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found. Running without custom CSS.")

#Title
st.markdown("""
<div class="card">
            <h1>LINEAR REGRESSION</h1>
            <p>Pedict <b>Tip Amount</b> from the <b>Total Bill </b>using Linear Regression..</p>
            </div>

""",unsafe_allow_html=True)

#Load Data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()
#Dataset preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("DataSet Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)
#Prepare Data
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#Train the model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#Metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
r2_score=r2_score(y_test,y_pred)
adj_r2=1-(1-r2_score)*(len(y_test)-1)/(len(y_test)-2)
#visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bills vs Tip")
fig,ax=plt.subplots()
ax.scatter(df['total_bill'],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total Bill ")
ax.set_ylabel("Tip ")
st.pyplot(fig)
st.markdown('</div',unsafe_allow_html=True)
#perfomance metrics
st.markdown('<div class="card">',unsafe_allow_html=True)
st.header("Model Perfomance")
c1,c2,c3=st.columns(3)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3.metric("MSE",f"{mse:.2f}")
c4,c5,c6=st.columns(3)
c4.metric("R2",f"{r2_score:.3f}")
c5.metric("Adjusted R2",f"{adj_r2:.3f}")
st.markdown('</div>',unsafe_allow_html=True)
#m&c
st.markdown(f"""
        <div class="card">
        <h3>Model Interception</h3>
        <p><b>co-efficent:</b>{model.coef_[0]:.3f}</br>
        <b>Intercept:</b>{model.intercept_:.3f}</p>
</div>
""",unsafe_allow_html=True)

#Prediction
# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: {tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

