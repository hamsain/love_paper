import os
from databases import Database

DB_HOST = os.getenv("MYSQL_HOST", "db") 

DATABASE_URL = f"mysql+aiomysql://root:mypass@{DB_HOST}/genai"

database = Database(DATABASE_URL)