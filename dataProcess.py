import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

# Load the medical dataset
file_path = 'healthcare-dataset-stroke-data.csv'
data = pd.read_csv(file_path)

data_loan = pd.read_csv("loan_data.csv")


######Generate synthetic data for loan_data.csv########
# Initialize metadata for a single table
metadata = Metadata()

# Add a table to the metadata
metadata.add_table(
    table_name='loan_data',  # Name of the table
)

# Add columns to the table
metadata.add_column(table_name='loan_data', column_name='Loan_ID', sdtype='id')
metadata.add_column(table_name='loan_data', column_name='Gender', sdtype='categorical')
metadata.add_column(table_name='loan_data', column_name='Married', sdtype='categorical')
metadata.add_column(table_name='loan_data', column_name='Dependents', sdtype='categorical')
metadata.add_column(table_name='loan_data', column_name='Education', sdtype='categorical')
metadata.add_column(table_name='loan_data', column_name='Self_Employed', sdtype='categorical')
metadata.add_column(table_name='loan_data', column_name='ApplicantIncome', sdtype='numerical')
metadata.add_column(table_name='loan_data', column_name='CoapplicantIncome', sdtype='numerical')
metadata.add_column(table_name='loan_data', column_name='LoanAmount', sdtype='numerical')
metadata.add_column(table_name='loan_data', column_name='Loan_Amount_Term', sdtype='numerical')
metadata.add_column(table_name='loan_data', column_name='Credit_History', sdtype='categorical')
metadata.add_column(table_name='loan_data', column_name='Property_Area', sdtype='categorical')
metadata.add_column(table_name='loan_data', column_name='Loan_Status', sdtype='categorical')

# Save metadata to a JSON file for replicability
#metadata.save_to_json('loan_metadata.json')

#load metadata from JSON file
metadata = Metadata.load_from_json('loan_metadata.json')

# Print metadata dictionary
print(metadata.to_dict())

# Initialize the synthesizer
copula = GaussianCopulaSynthesizer(metadata=metadata)

# Fit the synthesizer
copula.fit(data_loan)

# Sample synthetic data
num_synthetic_rows = 5200  # Reduced size to make debugging easier

synthetic_loan_data = copula.sample(num_synthetic_rows)

# Save the synthetic data to a new file and display it to the user
synthetic_file_path = 'synthetic_loan_data.csv'
synthetic_loan_data.to_csv(synthetic_file_path, index=False)

from sdv.evaluation.single_table import run_diagnostic

diagnostic = run_diagnostic(
    real_data=data_loan,
    synthetic_data=synthetic_loan_data,
    metadata=metadata
)


data2= pd.read_csv("loan_data.csv")

#make sure the data is shuffled
data = data.sample(frac=1, random_state=0).reset_index(drop=True)

# Convert the 'smoking_status' column to binary: 1 if 'formerly smoked' or 'smokes', 0 otherwise
data['smoking_status'] = data['smoking_status'].apply(lambda x: 1 if x in ['formerly smoked', 'smokes'] else 0)
data['Married'] = data['Married'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Self_Employed'] = data['Self_Employed'].apply(lambda x: 1 if x == 'Self-employed' else 0)
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
data['Property_Area'] = data['Property_Area'].apply(lambda x: 1 if x == 'Urban' else 0)


# Drop the 'id' column
data.drop(columns=['id'], inplace=True)
data.replace("", pd.NA, inplace=True)
data2.replace("", pd.NA, inplace=True)
#drop NA row
data.dropna(inplace=True)
data2.dropna(inplace=True)
#drop row if age is <18
data = data[data['age'] >= 18]

data2['Married'] = data2['Married'].apply(lambda x: 1 if x == 'Yes' else 0)
data2['Self_Employed'] = data2['Self_Employed'].apply(lambda x: 1 if x == 'Yes' else 0)
data2['Gender'] = data2['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
data2['Property_Area'] = data2['Property_Area'].apply(lambda x: 1 if x == 'Urban' else 0)


# Save the updated dataset to a new file and display it to the user
updated_file_path = 'medical_updated.csv'
data.to_csv(updated_file_path, index=False)

df1 = pd.DataFrame(data2)
df2 = pd.DataFrame(data)

# Merge based on 'id' and 'name'
merged_data = pd.merge(df1, df2, on=['Gender', 'Married','Self_Employed','Property_Area'], how='inner')

print(merged_data[['Loan_ID']].duplicated().sum())
merged_data = merged_data.drop_duplicates(subset=['Loan_ID'])


# Update the loan status to 'N' based on conditions
conditions = (
    (merged_data['heart_disease'] == 1) |
    (merged_data['bmi'] > 35) |
    (merged_data['smoking_status'] == 1)|
    (merged_data['stroke'] == 1)
)
merged_data['Loan_Status'] = merged_data['Loan_Status'].where(~conditions, 'N')


# Save the merged dataset to a new file and display it to the user
merged_file_path = 'merged_data.csv'
merged_data.to_csv(merged_file_path, index=False)




################## Synthetic Data Generation ##################

# Initialize metadata for a single table
metadata = Metadata()

# Add a table to the metadata
metadata.add_table(
    table_name='synthetic_data',  # Name of the table
)

# Add columns to the table
metadata.add_column(table_name='synthetic_data', column_name='Loan_ID', sdtype='id')
metadata.add_column(table_name='synthetic_data', column_name='Gender', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='Married', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='Dependents', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='Education', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='Self_Employed', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='ApplicantIncome', sdtype='numerical')
metadata.add_column(table_name='synthetic_data', column_name='CoapplicantIncome', sdtype='numerical')
metadata.add_column(table_name='synthetic_data', column_name='LoanAmount', sdtype='numerical')
metadata.add_column(table_name='synthetic_data', column_name='Loan_Amount_Term', sdtype='numerical')
metadata.add_column(table_name='synthetic_data', column_name='Credit_History', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='Property_Area', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='Loan_Status', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='age', sdtype='numerical')
metadata.add_column(table_name='synthetic_data', column_name='hypertension', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='heart_disease', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='avg_glucose_level', sdtype='numerical')
metadata.add_column(table_name='synthetic_data', column_name='bmi', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='smoking_status', sdtype='categorical')
metadata.add_column(table_name='synthetic_data', column_name='stroke', sdtype='categorical')

# Save metadata to a JSON file for replicability
#metadata.save_to_json('joint_metadata.json')

#load metadata from JSON file
metadata = Metadata.load_from_json('joint_metadata.json')

# Print metadata dictionary
print(metadata.to_dict())

# Initialize the synthesizer
copula = GaussianCopulaSynthesizer(metadata=metadata)

# Fit the synthesizer
copula.fit(merged_data)

# Sample synthetic data
num_synthetic_rows = 30000  # Reduced size to make debugging easier

synthetic_data = copula.sample(num_synthetic_rows)

# Save the synthetic data to a new file and display it to the user
synthetic_file_path = 'synthetic_data.csv'
synthetic_data.to_csv(synthetic_file_path, index=False)

from sdv.evaluation.single_table import run_diagnostic

diagnostic = run_diagnostic(
    real_data=merged_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

print(synthetic_data['stroke'].value_counts())
print(synthetic_data['Gender'].value_counts())
print(synthetic_data['Married'].value_counts())
print(synthetic_data['Self_Employed'].value_counts())
print(data_loan['Loan_Status'].value_counts())
print(data2['Loan_Status'].value_counts())
print(merged_data['Loan_Status'].value_counts())
print(synthetic_data['Loan_Status'].value_counts())
print(synthetic_data['smoking_status'].value_counts())
#print how many people have bmi > 35 and dont concatenate the data
print(data[data['bmi'] > 30].shape[0])










