import streamlit as st
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from openai import OpenAI
import os
from io import BytesIO


class MultilabelPredictor:
    multi_predictor_file = r"C:\Users\tewwa\Prompt_Eng\P1\agModels-predictEducationClass\multilabel_predictor.pkl"

    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, consider_labels_correlation=True, **kwargs):
        if len(labels) < 2:
            raise ValueError("MultilabelPredictor is only intended for predicting MULTIPLE labels (columns).")
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError("If provided, `problem_types` must have same length as `labels`")
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError("If provided, `eval_metrics` must have same length as `labels`")

        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}
        self.eval_metrics = {} if eval_metrics is None else {labels[i]: eval_metrics[i] for i in range(len(labels))}

        for i in range(len(labels)):
            label = labels[i]
            path_i = os.path.join(self.path, "Predictor_" + str(label))
            problem_type = problem_types[i] if problem_types is not None else None
            eval_metric = eval_metrics[i] if eval_metrics is not None else None
            self.predictors[label] = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=path_i, **kwargs)

    def fit(self, train_data, tuning_data=None, **kwargs):
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)

        train_data_og = train_data.copy()
        tuning_data_og = tuning_data.copy() if tuning_data is not None else None

        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))] if self.consider_labels_correlation else [l for l in self.labels if l != label]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            tuning_data = tuning_data_og.drop(labels_to_drop, axis=1) if tuning_data is not None else None
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
        self.save()

    def predict(self, data, **kwargs):
        return self._predict(data, as_proba=False, **kwargs)

    def predict_proba(self, data, **kwargs):
        return self._predict(data, as_proba=True, **kwargs)

    def evaluate(self, data, **kwargs):
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(path=os.path.join(self.path, self.multi_predictor_file), object=self)

    @classmethod
    def load(cls, path):
        return load_pkl.load(path=os.path.join(path, cls.multi_predictor_file))

    def get_predictor(self, label):
        predictor = self.predictors[label]
        return TabularPredictor.load(path=predictor) if isinstance(predictor, str) else predictor

    def _get_data(self, data):
        return TabularDataset(data) if isinstance(data, str) else data.copy()

    def _predict(self, data, as_proba=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(data, as_multiclass=True, **kwargs)
            data[label] = predictor.predict(data, **kwargs)
        return data[self.labels] if not as_proba else predproba_dict

# Function to process sales and customer data
def process_data(sale_file, customer_file):
    # Read sale data
    sale_df = pd.read_csv(sale_file)

    # Check if the first row is an extra header
    if sale_df.iloc[0].isnull().sum() == 0:
        sale_df.columns = sale_df.iloc[0]
        sale_df = sale_df[1:]
    sale_df.reset_index(drop=True, inplace=True)

    # Rename columns for consistency
    sale_df.rename(columns={'รหัสลูกค้า': 'Customer ID'}, inplace=True)

    # Read customer data
    cus_df = pd.read_excel(customer_file)

    # Merge sale and customer data
    merged_df = pd.merge(sale_df, cus_df, on='Customer ID', how='inner')
    merged_df.fillna(0, inplace=True)
    # Remove commas and parentheses from all values in filtered_df
    merged_df = merged_df.replace({',': '', r'\(': '', r'\)': ''}, regex=True)
    # st.dataframe(merged_df)
    if not all(col in merged_df.columns for col in ["Credit Term(Day)", "Credit Value"]):
        st.write("Predicting Credit Term(Day) and Credit Value...")
        labels = ['Credit Term(Day)','Credit Value']  # which columns to predict based on the others
        problem_types = ['multiclass', 'multiclass']  # type of each prediction problem (optional)
        eval_metrics = ['accuracy', 'accuracy']  # metrics used to evaluate predictions for each label (optional)
        # time_limit = 120
        
        multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, eval_metrics=eval_metrics)
        predictor = multi_predictor.load("Model.zip")
        # multi_predictor.fit(merged_df, time_limit=time_limit, presets='high_quality')
        predictions = predictor.predict(merged_df)
        
        merged_df["Credit Term(Day)"] = predictions["Credit Term(Day)"]
        merged_df["Credit Value"] = predictions["Credit Value"]
    
    merged_df = merged_df.astype({
    "มูลค่ารวมก่อนภาษี": float,
    "มูลค่า": float,
    "จำนวนเงินที่ชำระ": float,
    "จำนวน": float,
    "ราคาต่อหน่วย": float,
    "ราคารวม": float,
    "Credit Value": float,
    "Credit Term(Day)": float
})
    
    # After ensuring the correct data types, perform the groupby and aggregation
    customer_summary = merged_df.groupby('Customer ID').agg({
        'มูลค่ารวมก่อนภาษี': ['sum', 'mean'],      # Total Spending, Average, Variability
        'สถานะรายการ': lambda x: (x == 'สำเร็จ').mean(), # Percentage of successful payments
        'จำนวนเงินที่ชำระ': 'sum',
        'Type Of Customer': 'first',                       # Type of Customer
        'Credit Value': ['sum', 'mean'],                   # Total and Average Credit Value
        'Credit Term(Day)': 'mean',                        # Average Credit Term
        'Customer ID': 'count'                             # Frequency of Purchases
    })

    # Flatten the MultiIndex columns
    customer_summary.columns = ['_'.join(col).strip() for col in customer_summary.columns.values]

    # Rename columns for clarity
    customer_summary.rename(columns={
        'มูลค่ารวมก่อนภาษี_sum': 'Total_Spending',
        'มูลค่ารวมก่อนภาษี_mean': 'Average_Transaction_Amount',
        'สถานะรายการ_<lambda>': 'Successful_Payment_Rate',
        'จำนวนเงินที่ชำระ_sum': 'Total_Paid_Amount',
        'Credit Value_sum': 'Total_Credit_Value',
        'Credit Value_mean': 'Average_Credit_Value',
        'Credit Term(Day)_mean': 'Average_Credit_Term',
        'Customer ID_count': 'Frequency_of_Purchases',
        'Type Of Customer_first': 'Type_Of_Customer'
    }, inplace=True)

    # Reset index to make 'Customer ID' a column again
    customer_summary.reset_index(inplace=True)
    return merged_df, customer_summary
    
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

# # Set your OpenAI GPT API key
client = OpenAI(
#    api_key=API_KEY
)

# Function for GPT-based risk assessment
def assess_risk(customer_summary_row):

    customer_details = customer_summary_row.to_frame().transpose().to_string(index=False)
    # Call GPT API for risk assessment
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use an appropriate model available to you
            messages=[
                {"role": "system", "content": f'''You are an expert in credit risk assessment.
                Determine the risk level (ต่ำ, กลาง, สูง). 0 average credit value and 0 average credit term(Day) mean the customer has no debt which is good. You think step by step. Clearly explain your answer based on the following details: {customer_details}.
                You must respond in Thai. Format your response as: ความเสี่ยง : เหตุผล '''},
                {"role": "user", "content": "Determine the risk level (ต่ำ, กลาง, สูง)."}
            ]
        )

        # Return GPT-API response
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during risk assessment: {e}")
        return None

def save_to_excel_all(customers_data):
    output = BytesIO()
    
    # Save to memory (for Streamlit download)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        customers_data.to_excel(writer, sheet_name='Risk Assessment', index=False)
    
    # Rewind the BytesIO stream to the beginning for download in Streamlit
    output.seek(0)
    return output

def save_to_excel_single(customer_id, customer_type, total_spending, avg_transaction, risk_level, customer_info):
    output = BytesIO()
    
    # Save to memory (for Streamlit download)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df = pd.DataFrame({
            'Customer ID': [customer_id],
            'Customer Type': [customer_type],
            'Total Spending': [total_spending],
            'Average Transaction Amount': [avg_transaction],
            'Risk Level': [risk_level]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        customer_info[['มูลค่ารวมก่อนภาษี', 'สถานะรายการ']].to_excel(writer, sheet_name='Customer Activities', index=False)
    
    # Rewind the BytesIO stream to the beginning for download in Streamlit
    output.seek(0)
    return output

# Streamlit App
st.title("Customer Risk Assessment Tool")

# File Uploads
sale_file = st.file_uploader("Upload Sale Data (CSV)", type=["csv"])
customer_file = st.file_uploader("Upload Customer Data (Excel)", type=["xlsx"])

if sale_file and customer_file:
    st.write("Processing Data...")
    merged_df, customer_summary = process_data(sale_file, customer_file)

    if merged_df is None or customer_summary is None:
        st.stop()

    st.write("Customer Summary:")
    st.dataframe(customer_summary)

    # Input for customer ID
    customer_id = st.text_input("Enter Customer ID")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Assess Risk Level for Single Customer"):
            customer_info = merged_df[merged_df['Customer ID'] == customer_id]
            if customer_info.empty:
                st.error(f"Customer ID '{customer_id}' not found.")
            else:
                total_spending = customer_info['มูลค่ารวมก่อนภาษี'].sum()
                avg_transaction = customer_info['มูลค่ารวมก่อนภาษี'].mean()
                total_paid_amount = customer_info['จำนวนเงินที่ชำระ'].sum()
                customer_type = customer_info['Type Of Customer'].iloc[0] if 'Type Of Customer' in customer_info.columns else 'Unknown'
                customer_summary_row = customer_summary[customer_summary['Customer ID'] == customer_id].iloc[0]
                risk_assessment = assess_risk(customer_summary_row)

                st.write(f"Risk Assessment: {risk_assessment}")

                # Store variables in session_state
                st.session_state['customer_id'] = customer_id
                st.session_state['customer_type'] = customer_type
                st.session_state['total_spending'] = total_spending
                st.session_state['avg_transaction'] = avg_transaction
                st.session_state['total_paid_amount'] = total_paid_amount
                st.session_state['risk_assessment'] = risk_assessment
                st.session_state['customer_info'] = customer_info

    with col2:
        if st.button("Assess Risk Level for All Customers"):
            st.write("Assessing risk for all customers. This may take a while...")
            all_customers = customer_summary.copy()
            all_customers['Risk Assessment'] = all_customers.apply(assess_risk, axis=1)
            st.session_state['all_customers'] = all_customers
            st.write("Risk assessment for all customers completed.")
            st.dataframe(all_customers)

    # Download button for single customer
    if 'customer_id' in st.session_state:
        excel_file = save_to_excel_single(
            customer_id=st.session_state['customer_id'],
            customer_type=st.session_state['customer_type'],
            total_spending=st.session_state['total_spending'],
            avg_transaction=st.session_state['avg_transaction'],
            risk_level=st.session_state['risk_assessment'],
            customer_info=st.session_state['customer_info']
        )

        st.download_button(
            label="Download Single Customer Results",
            data=excel_file,
            file_name=f"customer_{st.session_state['customer_id']}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Download button for all customers
    if 'all_customers' in st.session_state:
        excel_file_all = save_to_excel_all(st.session_state['all_customers'])

        st.download_button(
            label="Download All Customers Results",
            data=excel_file_all,
            file_name="all_customers_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
