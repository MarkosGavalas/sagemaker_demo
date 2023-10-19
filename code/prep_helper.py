
def improve_column_val(df):
    value_mapper = {'Female': 'F', 'Male': 'M', '1':'Y', '0':'N',
                    'No phone service': 'No_phone', 'Fiber optic': 'Fiber',
                    'No internet service': 'No_internet', 'Month-to-month': 'Monthly',
                    'Bank transfer (automatic)': 'Bank_transfer',
                    'Credit card (automatic)': 'Credit_card',
                    'One year': 'One_year', 'Two year': 'Two_years', 
                    'Electronic check':'Electronic_check', 'Mailed check': 'Mailed_check'}
    df.replace(to_replace=value_mapper, inplace=True)
    return df
