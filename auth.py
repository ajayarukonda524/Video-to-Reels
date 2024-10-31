import streamlit as st
import psycopg2
import logging
import os
import bcrypt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "Video_to_Reels_UA"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "2004")
        )
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        st.error("Could not connect to the database. Please check your configuration.")
        return None

# Password hashing with bcrypt
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Password verification
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def signup(first_name, last_name, username, password, occupation, email, phone_number, profile_picture):
    hashed_password = hash_password(password)
    conn = get_db_connection()
    
    if conn is None:
        return  # Exit if database connection failed

    try:
        cur = conn.cursor()
        cur.execute(""" 
            INSERT INTO users (first_name, last_name, username, password, occupation, email, phone_number, profile_picture) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (first_name, last_name, username, hashed_password, occupation, email, phone_number, profile_picture))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except psycopg2.IntegrityError:
        conn.rollback()
        logging.warning("Integrity error on signup: Username already exists.")
        st.error("Username already exists. Please choose a different one.")
    except Exception as e:
        logging.error(f"Signup error: {str(e)}")
        st.error("An error occurred during signup. Please try again later.")
    finally:
        cur.close()
        conn.close()


# Check credentials for login
def check_credentials(username, password):
    conn = get_db_connection()
    
    if conn is None:
        return "Database connection failed."  # Exit if database connection failed

    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        if user and check_password(password, user[2]):  # Assuming password hash is in the 4th column
            return user  # user will now include the profile picture URL
        else:
            logging.info("Invalid login attempt.")
            return None
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return None

    finally:
        conn.close()
