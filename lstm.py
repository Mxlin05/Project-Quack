import tensorflow
import pandas as pd
import urllib.parse
import sqlalchemy as sqla
import dotenv
import os

#Gets login data to access database
dotenv.load_dotenv("database.env")

#Connecting to the database
conn_str = (
    "DRIVER=ODBC Driver 18 for SQL Server;"
    f"SERVER={os.getenv('Server')};"
    "DATABASE=Project Quack;"
    f"UID={os.getenv('UserId')};"
    f"PWD={os.getenv('Password')};"
    "TrustServerCertificate=yes;"
)
conn_url = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
url_object = urllib.parse.urlparse(conn_url)
engine = sqla.create_engine(
    conn_url,
    fast_executemany=True,  
    connect_args={'timeout': 30} 
)
