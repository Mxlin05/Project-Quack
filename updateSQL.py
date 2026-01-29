print ("Started")
import pandas as pd
import urllib
import sqlalchemy as sqla
import dotenv
import os

'''
Updating sql database
1. Download the libraries (py -m pip install sqlalchemy pyodbc dotenv)
2. Download ODBC driver 18 from microsoft
'''

#Gets login data to access database
dotenv.load_dotenv("database.env")

test_df = pd.DataFrame({
    "Names": ["Person1", "Person2", "Person3"],
    "Ages": [15, 23, 67]
})

#Connecting to the database
conn_str = (
    "DRIVER=ODBC Driver 18 for SQL Server;"
    f"SERVER={os.getenv("Server")};"
    "DATABASE=Project Quack;"
    f"UID={os.getenv("UserId")};"
    f"PWD={os.getenv("Password")}"
)
conn_url = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
engine = sqla.create_engine(conn_url)

#Uploading Dataframe to database
test_df.to_sql(
    "test",
    engine,
    index=False,
    if_exists="replace"
)

#Reading Dataframe from database
returned_data = pd.read_sql("SELECT * FROM test", engine)
print(returned_data)