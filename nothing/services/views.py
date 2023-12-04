import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.staticfiles import finders
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Create your views here.

def service(request):
    return render(request, 'service.html')


def loan_algorithm(data,custom_input):
    # Dropping Loan_ID column because there is no use for prediction
    data.drop(['Loan_ID'], axis=1, inplace=True)

    # encoding categorical columns into int
    obj = (data.dtypes == 'object')
    label_encoder = LabelEncoder()
    obj = (data.dtypes == 'object')
    for col in list(obj[obj].index):
        data[col] = label_encoder.fit_transform(data[col])

    # taking action on missing values in dataset
    for col in data.columns:
        data[col] = data[col].fillna(data[col].mean())

    # extract dependent and independent variable in dataset
    X = data.drop(['Loan_Status'], axis=1)
    Y = data['Loan_Status']

    # scaling is important to improve accuracy of model but,
    # in case of random forest algorithmRandom Forests are robust to,
    # variations in feature scales

    # spliting dataset into training and testing dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # create instance of random forest model
    random_forest_classifier = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)

    # train model using random forest
    random_forest_classifier.fit(X, Y)

    # make prediction :)
    # Convert the custom input to a DataFrame
    custom_input_df = pd.DataFrame(custom_input,index=[0])
    print(custom_input_df.head())

    result = random_forest_classifier.predict(custom_input_df)

    # accuracy of random forest
    # accuracy = accuracy_score(Y_test,result)*100
    # accuracy = random_forest_classifier.score(X_test,Y_test)*100    #0.9586776859504132

    return result


def home_algorithm(df,custome_input):

    def grp_loc(locality):
        locality = locality.lower()
        if 'rohini' in locality:
            return 'Rohini Sector'
        elif 'dwarka' in locality:
            return 'Dwarka Sector'
        elif 'shahdara' in locality:
            return 'Shahdara'
        elif 'vasant' in locality:
            return 'Vasant Kunj'
        elif 'paschim' in locality:
            return 'Paschim Vihar'
        elif 'alaknanda' in locality:
            return 'Alaknanda'
        elif 'vasundhara' in locality:
            return 'Vasundhara Enclave'
        elif 'punjabi' in locality:
            return 'Punjabi Bagh'
        elif 'kalkaji' in locality:
            return 'Kalkaji'
        elif 'lajpat' in locality:
            return 'Lajpat Nagar'
        elif 'laxmi' in locality:
            return 'Laxmi Nagar'
        elif 'patel' in locality:
            return 'Patel Nagar'
        else:
            return 'Other'

    df['Locality'] = df['Locality'].apply(grp_loc)
    encoder = LabelEncoder()
    col = ['Furnishing', 'Locality', 'Status', 'Transaction', 'Type']
    for i in col:
        encoder.fit(df[i])
        df[i] = encoder.transform(df[i])
        print(i, df[i].unique())
    df2 = df.copy()
    df2.drop(['Bathroom', 'Parking', 'Status', 'Transaction', 'Type', 'Per_Sqft'], inplace=True, axis=1)
    print(df2.head())
    # splitting dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(df2.drop('Price', axis=1), df2.Price, test_size=0.2,
                                                        random_state=0)
    # Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    custome_input_df = pd.DataFrame(custome_input,index=[0])
    print(custome_input_df)
    result = model.predict(custome_input_df)
    accuracy = model.score(X_train,y_train)
    return result



def loan(request):
    if request.method == 'POST':
        gender_mapping = {'Female': 0, 'Male': 1}
        married_mapping = {'No': 0, 'Yes': 1}
        education_mapping = {'Graduate': 0, 'Not Graduate': 1}
        self_employed_mapping = {'No': 0, 'Yes': 1}
        property_area_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

        gender = request.POST.get('gender')
        married = request.POST.get('married')
        dependents = request.POST.get('dependents')
        education = request.POST.get('education')
        self_employed = request.POST.get('self_employed')
        applicant_income = request.POST.get('applicant_income')
        coapplicant_income = request.POST.get('coapplicant_income')
        loan_amount = request.POST.get('loan_amount')
        loan_term = request.POST.get('loan_term')
        credit_history = request.POST.get('credit_history')
        property_area = request.POST.get('property_area')

        userdata = {
            'Gender': [gender_mapping.get(gender)],
            'Married': [married_mapping.get(married)],
            'Dependents': [dependents],
            'Education': [education_mapping.get(education)],
            'Self_Employed': [self_employed_mapping.get(self_employed)],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area_mapping.get(property_area)]
        }



        # Resolve the file path
        file_path = finders.find('dataset/LoanApprovalPrediction.csv')

        # Load the dataset
        data = pd.read_csv(file_path)
        result = loan_algorithm(data, userdata)
        print(result)
        userdata['Loan_Status'] = result
        return render(request, 'result.html', {'userdata': userdata})
        print(userdata)
        request.session['userdata'] = userdata
        return render(request,'result.html')
    else:
        return render(request, 'loan.html')


def house(request):
    if request.method == 'POST':
        area = int(request.POST.get('area'))
        bhk = int(request.POST.get('bhk'))
        furnishing = int(request.POST.get('furnished'))
        locality = int(request.POST.get('locality'))

        userdata1 = {
            'Area': [area],
            'BHK': [bhk],
            'Furnishing': [furnishing],
            'Locality': [locality],
        }
        print(userdata1)
        file_path1 = finders.find('dataset/MagicBricks.csv')

        # Load the dataset
        df = pd.read_csv(file_path1)
        result = home_algorithm(df, userdata1)
        print(result)
        userdata1['House_Status'] = result
        return render(request, 'house_result.html', {'userdata': userdata1})
        request.session['userdata'] = userdata1
        return render(request, 'house_result.html')
    else:
        return render(request, 'house.html')