from databases import Database

DATABASE_URL = "mysql+aiomysql://root:mypass@localhost:3307/genai"

database = Database(DATABASE_URL)