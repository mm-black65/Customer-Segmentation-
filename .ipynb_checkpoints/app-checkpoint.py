#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
import joblib


# In[4]:


kmeans = joblib.load("kmeans_model.pkl")


# In[5]:


def label_cluster(cluster):
    if cluster == 0:
        return "High Income - High Spending (VIP)"
    elif cluster == 1:
        return "Low Income - Low Spending"
    elif cluster == 2:
        return "High Income - Low Spending (Saver)"
    elif cluster == 3:
        return "Low Income - High Spending"
    else:
        return "Average Customer"


# In[6]:


st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️")
st.title("🛍️ Customer Segmentation")
st.write("Enter customer details to identify their segment.")


# ## Input section

# In[7]:


st.subheader("Enter Customer Details")

income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)


# In[8]:


if st.button("Predict Customer Type"):
    input_data = np.array([[income, spending]])

    cluster = kmeans.predict(input_data)[0]

    result = label_cluster(cluster)

    st.subheader("Result")
    st.success(result)

    st.write(f"Cluster: {cluster}")


# In[ ]:




