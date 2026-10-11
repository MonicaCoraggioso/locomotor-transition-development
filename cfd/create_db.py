# %% This script initializes a SQLite database and creates a table for storing results.

from pathlib import Path
import sqlite3
import logging

from utils import get_root_folder

logging.basicConfig(level=logging.INFO)

root_folder = get_root_folder()

# %%
# Define the column names and their Python types
dtype = {
    "dpf": str,
    "fish": str,
    "start_frame": int,
    "end_frame": int,
    "exp_speed": float,
    "cfd_speed": float,
    "RE (%)": float,
    "exp_distance": float,
    "cfd_distance": float,
    "ibamr_energy": float,
    "bout_type": str,
}

# Path to the SQLite database
db_file = root_folder / "results.db"

# Connect and create table if not exists
conn = sqlite3.connect(db_file)
column_defs = []
for col, py_type in dtype.items():
    if py_type is str:
        sql_type = "TEXT"
    elif py_type is int:
        sql_type = "INTEGER"
    elif py_type is float:
        sql_type = "REAL"
    else:
        sql_type = "BLOB"
    column_defs.append(f'"{col}" {sql_type}')

create_sql = f'''
CREATE TABLE IF NOT EXISTS results (
    {", ".join(column_defs)}
)
'''

with conn:
    conn.execute(create_sql)

conn.close()
logging.info(f"Initialized database and ensured 'results' table exists at {db_file}")
