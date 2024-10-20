import streamlit as st
import psycopg2
import hashlib
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",  # Change as needed
        database="Video_to_Reels_UA",  # Change as needed
        user="postgres",  # Change as needed
        password="ajay@524",  # Change as needed
        port="2004"
    )
    return conn

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Signup function with additional details
def signup(first_name, last_name, username, password, occupation, email, phone_number):
    hashed_password = hash_password(password)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(""" 
            INSERT INTO users (first_name, last_name, username, password, occupation, email, phone_number) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (first_name, last_name, username, hashed_password, occupation, email, phone_number))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except psycopg2.IntegrityError:
        conn.rollback()
        st.error("Username already exists. Please choose a different one.")
    finally:
        cur.close()
        conn.close()

# Check credentials for login
def check_credentials(username, password):
    hashed_password = hash_password(password)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, hashed_password))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user  # Returns user details if correct, otherwise None
