import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# read the available datasets
tr = pd.read_csv("train.csv")
tst = pd.read_csv("test.csv")

# Histogram of applicant income
tr['ApplicantIncome'].hist(bins=50)
plt.ylabel('No. of Applicants')
plt.xlabel('ApplicantIncome')
plt.show()

# Histogram of Loan amount
tr['LoanAmount'].hist(bins=50)
plt.ylabel('No. of Applicants')
plt.xlabel('LoanAmount')
plt.show()

# boxplot of loan amount
tr.boxplot(column='LoanAmount')
plt.show()

# boxplot of Applicant Income by education
tr.boxplot(column='ApplicantIncome', by='Education')
plt.show()

tr['Property_Area'].value_counts()

# save the no. of observations in training and test dataset
tr_len = len(tr)
tst_len = len(tst)

tr['Type'] = 'Train'
tst['Type'] = 'Test'

data_all = pd.concat([tr, tst], axis=0, sort=True)
data_all['LoanAmount_log'] = np.log(data_all['LoanAmount'])

# Process the data to fill in all missing values
data_all['LoanAmount'].fillna(data_all['LoanAmount'].mean(), inplace=True)
data_all['LoanAmount_log'].fillna(data_all['LoanAmount_log'].mean(), inplace=True)
data_all['Loan_Amount_Term'].fillna(data_all['Loan_Amount_Term'].mean(), inplace=True)
data_all['ApplicantIncome'].fillna(data_all['ApplicantIncome'].mean(), inplace=True)
data_all['CoapplicantIncome'].fillna(data_all['CoapplicantIncome'].mean(), inplace=True)
data_all['Gender'].fillna(data_all['Gender'].mode()[0], inplace=True)
data_all['Married'].fillna(data_all['Married'].mode()[0], inplace=True)
data_all['Dependents'].fillna(data_all['Dependents'].mode()[0], inplace=True)
data_all['Loan_Amount_Term'].fillna(data_all['Loan_Amount_Term'].mode()[0], inplace=True)
data_all['Credit_History'].fillna(data_all['Credit_History'].mode()[0], inplace=True)

categories = ['Credit_History', 'Dependents', 'Gender', 'Married', 'Education', 'Property_Area', 'Self_Employed',
              'LoanAmount', 'Credit_History']

# convert data in numeric form
for x in categories:
    n = LabelEncoder()
    data_all[x] = n.fit_transform(data_all[x].astype('str'))
tr.dtypes

# add up the applicant income and co-applicant income data as both will be considers together
data_all['TotalIncome'] = data_all['ApplicantIncome'] + data_all['CoapplicantIncome']
data_all['TotalIncome_log'] = np.log(data_all['TotalIncome'])
data_all['LoanAmount_log'] = np.log(data_all['LoanAmount'])

# Histogram for Total Income
data_all['TotalIncome_log'].hist(bins=20)


# Classifier to predict output and calculate accuracy
def classification(md, dt_final, pred, out):
    pr = dt_final[pred].values
    dt = dt_final[out]
    md.fit(pr, dt)
    predictions = md.predict(dt_final[pred])

    # Display the model accuracy
    accuracy = metrics.accuracy_score(predictions, dt_final[out])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    print("Confusion Matrix on Test Data")
    confusion_matrix = pd.crosstab(dt_final[out], predictions, rownames=['Actual'], colnames=['Predicted'],
                                   margins=True)
    print(confusion_matrix)


tr_mod = data_all[data_all['Type'] == 'Train']
tst_mod = data_all[data_all['Type'] == 'Test']
tr_mod["Loan_Status"] = n.fit_transform(tr_mod["Loan_Status"].astype('str'))

# select the parameters from the training dataset that will be used to predict
sel_predictors = ['Credit_History', 'Education', 'Married', 'TotalIncome_log', 'LoanAmount']

x_train = tr_mod[list(sel_predictors)].values
y_train = tr_mod["Loan_Status"].values
x_test = tst_mod[list(sel_predictors)].values

# Train the Logistic regression classification model
model = LogisticRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)
predict = n.inverse_transform(predict)
tst_mod['Loan_Status'] = predict
out_x = 'Loan_Status'
print("Logistic Regression")
classification(model, tr_mod, sel_predictors, out_x)

# Train the Decision Tree classification model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
predictions = n.inverse_transform(predictions)
tst_mod['Loan_Status'] = predictions
out_x = 'Loan_Status'
print("Decision Tree Classifier")
classification(model, tr_mod, sel_predictors, out_x)
