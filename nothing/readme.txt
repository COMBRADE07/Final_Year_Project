project requirements:
1.python
2.sklearn
3.django
4.mysqlclient
5.MySQL/ xampp
6.create db 'nothing'



--------------------------
csv to sql converter
-replace your file name,table and databse
--------------------------
import pandas as pd
from sqlalchemy import create_engine

def csv_to_mysql(csv_file, table_name, database_url):
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Create a MySQL connection using SQLAlchemy
    engine = create_engine(database_url)

    # Write DataFrame to MySQL database
    df.to_sql(name=table_name, con=engine, index=False, if_exists='replace')

# Specify your CSV file, MySQL database URL, and table name
csv_file_path = 'LoanApprovalPrediction.csv'
mysql_database_url = 'mysql://root:@localhost:3306/nothing'
table_name = 'loan'

# Call the function
csv_to_mysql(csv_file_path, table_name, mysql_database_url)