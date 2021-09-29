import streamlit as st

#Import relevant libararies.
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


#Import libraries for Association rule mining
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

@st.cache
def get_data(filename):
    restaurant_data = pd.read_csv(filename)
    return restaurant_data
hide_menu_style ="""
        <style>
            #MainMenu {
                visibility: hidden;
            }
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

#load the model from disk
import pickle
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

def predict_note_autentication(seniorcitizen,dependents,tenure,contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges,
mutliplelines,phoneservice,internetservice,onlinesecurity,onlinebackup,deviceprotection,techsupport):

    if seniorcitizen == 'Yes':
        seniorcitizen = 1
    else:
        seniorcitizen = 0

    if dependents == 'Yes':
        dependents = 1
    else: 
        dependents = 0

    if contract == 'Month-to-month':
        contract = 0
    elif contract == 'One year':
        contract = 1
    else:
        contract = 2

    if paperlessbilling == 'Yes':
        paperlessbilling = 1
    else:
        paperlessbilling = 0

    if paymentmethod == 'Bank transfer (automatic)':
        paymentmethod = 0
    elif paymentmethod == 'Credit card (automatic)':
        paymentmethod = 1 
    elif paymentmethod == 'Electronic check':
        paymentmethod = 2
    else:
        paymentmethod = 3

    if mutliplelines == 'No':
        mutliplelines = 0
    elif mutliplelines == 'No phone service':
        mutliplelines = 1
    else:
        mutliplelines = 2

    if phoneservice == 'No':
        phoneservice = 0
    else:
        phoneservice = 1

    if internetservice == 'DSL':
        internetservice = 0
    elif internetservice == 'Fiber optic':
        internetservice = 1
    else:
        internetservice = 2

    if onlinesecurity == 'No':
        onlinesecurity = 0
    elif onlinesecurity == 'No internet service':
        onlinesecurity = 1
    else:
        onlinesecurity = 2

    if onlinebackup == 'No':
        onlinebackup = 0
    elif onlinebackup == 'No internet service':
        onlinebackup = 1
    else:
        onlinebackup = 2

    if deviceprotection == 'No':
        deviceprotection = 0
    elif deviceprotection == 'No internet service':
        deviceprotection = 1
    else:
        deviceprotection = 2

    if techsupport == 'No':
        techsupport = 0
    elif techsupport == 'No internet service':
        techsupport = 1
    else:
        techsupport = 2

    #Predictions
    prediction = classifier.predict([[seniorcitizen,dependents,tenure,contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges,
                 mutliplelines,phoneservice,internetservice,onlinesecurity,onlinebackup,deviceprotection,techsupport]])

    if prediction == 0:
        pred = 'not terminated'
    else:
        pred = 'terminated'

    return pred


def main():

    page = st.sidebar.selectbox("Go To", ["Association", "Classification"])

    #Association Rule Mining
    if page == "Association":

        data_set = st.container()
        plots = st.container()

        with data_set:

            
            st.markdown(
            """
                <style> 
                    .main { 
                        background-image: url("https://wowfood.guru/templates/Default/images/intro1.jpg"); 
                        background-size: cover; } 
                </style>
            """,
             unsafe_allow_html=True
            )
            st.markdown('<h1 style="float:left;color:white;text-align:left;">Curry Kingdom Restaurant</h1>', unsafe_allow_html=True)
            
            #Import the Dataset 
            dataSet = get_data('restaurant-1-orders.csv')
            items = dataSet["Item Name"].unique()
            items_selected = st.multiselect('Select Item Set', items)

            Order_List = []

            for i in dataSet["Order Number"].unique():
                List = list(set(dataSet[dataSet["Order Number"]==i]["Item Name"]))
                if len(List)>0:
                    Order_List.append(List)
                
            te = TransactionEncoder()
            te_array = te.fit(Order_List).transform(Order_List)
            data_Set1 = pd.DataFrame(te_array, columns= te.columns_)

            # Generate frequent itemsets
            freq_items = apriori(data_Set1, min_support=0.01, use_colnames=True)

            # Generate Rules
            rules = association_rules(freq_items, metric="lift", min_threshold = 1.0)

            # Filter the dataset for a lift is greater than or equal 1 and confidence is grater than or equal 50%
            freq_rules = rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.5)].sort_values(['confidence'], ascending = True)

            antecedent_rules = freq_rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

            #st.write(antecedent_rules)

            items = frozenset(items_selected)

            l = 0

            if st.button("Show Consequent Items"):
                a = []
                for i in freq_rules['antecedents']:
                    l = l + 1
                    #st.write(i)
                    if i == items:
                        consequentItems = freq_rules[freq_rules["antecedents"]==i]["consequents"]
                        consequents = list(consequentItems)

                        a = [list(x) for x in consequents]
                        #b = ('\n'.join(map(str,a)))
                        #break

                if len(a) > 0:
                    for elm1 in a:
                        if len(elm1) == 1:
                            elm = elm1[0]
                            st.error(elm)
                            st.write("") 
                            st.write("")  
                            st.write("")
                        else:
                            for elm2 in elm1:
                                st.error(elm2)
                            st.write("")
                            st.write("")
                            st.write("")        
                else:
                    st.error("No Consequent Items...")

                    
            


    #Classification using Logistic Regression 
    elif page == "Classification":

        original_title = '<h6 style="font-family:Georgia, serif; color:White; font-size:38px; text-align:center;"> Telco Customer Churn Prediction App </h6>'
        
        st.markdown(
            """
                <style> 
                    .main { 
                        background-image: url("https://img.rawpixel.com/s3fs-private/rawpixel_images/creative_room/v796-nunny-03b_2.jpg?w=1000&dpr=1&fit=default&crop=default&q=65&vib=3&con=3&usm=15&bg=F4F4F3&ixlib=js-2.2.1&s=ace5aea9ad64b2c48c278bdc8c31773d"); 
                        background-size: cover;
                        color:white } 
                    }
                </style>
            """,
             unsafe_allow_html=True
            )
        st.markdown(original_title, unsafe_allow_html=True)

        st.markdown("""
            Predict customer churn based on the past records.
            The application is functional for both Online prediction and Batch data prediction.
        """)
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        st.markdown(
        """
            <style> 
                .main1 { 
                   
                    color:white } 
                }
            </style>
        """,
         unsafe_allow_html=True
        )
        
        #features selection
        st.subheader("Customer Details")
        seniorcitizen = st.selectbox('Senior Citizen : ', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)

        st.subheader("Services signed up for")
        phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
        mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
        deviceprotection = st.selectbox("Does the customer have device protection",('Yes','No','No internet service'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))

        st.subheader("Payment Details")
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        paymentmethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0.00, max_value=150.00, value=0.00)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0.00, max_value=10000.00, value=0.00)

        data = {
                'SeniorCitizen': seniorcitizen,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'DeviceProtection': deviceprotection,
                'TechSupport': techsupport,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':paymentmethod, 
                'MonthlyCharges': monthlycharges, 
                'TotalCharges': totalcharges
        }

        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        result = ""

        if st.button('Predict'):
            result = predict_note_autentication(seniorcitizen,dependents,tenure,contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges,
                    mutliplelines,phoneservice,internetservice,onlinesecurity,onlinebackup,deviceprotection,techsupport)


            st.error('The Telco Service is {}'.format(result))


if __name__ == '__main__' :
    main()

        