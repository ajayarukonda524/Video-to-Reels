import streamlit as st
import auth 
import whisper
import ffmpeg
import openai
import os
import json
import logging
from tqdm import tqdm
import sys

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logging.error("OpenAI API key not set in environment variables.")
    st.error("OpenAI API key is required. Please set it in your environment.")
    st.stop()

# Function to extract audio from video
def extract_audio(video_path, output_audio_path):
    ffmpeg.input(video_path).output(output_audio_path).run()

# Transcribing Audio to Text using OpenAI Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text'], result['segments']

# Analyzing Text Segments for Importance using OpenAI API
@st.cache_data
def analyze_text_importance(segments):
    important_segments = []
    for segment in tqdm(segments, desc="Analyzing segments"):
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Please rate the importance of the following text on a scale of 0 to 10: {text}"}],
            max_tokens=10,
            temperature=0
        )

        try:
            importance_score = float(response['choices'][0]['message']['content'].strip())
        except (ValueError, IndexError):
            logging.error(f"Unexpected response: {response['choices'][0]['message']['content']}")
            importance_score = 0.0  # Default score if parsing fails

        if importance_score > 1.0:
            important_segments.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'importance_score': importance_score
            })
    return important_segments

# Save important segments to file
def save_segments_to_file(segments, filename='important_segments.json'):
    with open(filename, 'w') as f:
        json.dump(segments, f, indent=4)

# Extracting Video Segments Based on Timestamps
def extract_video_segment(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    ffmpeg.input(video_path, ss=start_time, t=duration).output(output_path).run()

# Compiling Extracted Segments into a 30-Second Reel
def compile_video_segments(segment_paths, audio_path, output_video_path):
    try:
        # Create input streams for video segments
        video_inputs = [ffmpeg.input(segment) for segment in segment_paths]

        # Concatenate video segments using the concat filter
        video_stream = ffmpeg.concat(*video_inputs, v=1, a=0).node  # Concatenate video, no audio
        audio_stream = ffmpeg.input(audio_path)
        output = ffmpeg.output(video_stream[0], audio_stream, output_video_path, c='copy', c_a='aac')

        # Run the FFmpeg command
        output.run(overwrite_output=True)

    except ffmpeg.Error as e:
        logging.error(f'FFmpeg error: {e.stderr.decode("utf-8") if e.stderr else "Unknown error"}')
        st.error("An error occurred during video processing. Check logs for details.")

# UI for generating reel from important segments
def generate_reel_from_important_segments(video_path):
    audio_path = 'output_audio.mp3'
    extract_audio(video_path, audio_path)
    _, segments = transcribe_audio(audio_path)
    important_segments = analyze_text_importance(segments)
    important_segments.sort(key=lambda x: x['importance_score'], reverse=True)
    top_segments = important_segments[:3]  # Automatically take the top 3 segments for the reel

    save_segments_to_file(top_segments, 'important_segments.json')

    segment_paths = []
    for i, segment in enumerate(top_segments):
        start_time = segment['start_time']
        end_time = segment['end_time']
        output_path = f'segment_{i + 1}.mp4'
        extract_video_segment(video_path, start_time, end_time, output_path)
        segment_paths.append(output_path)

    compiled_video_path = 'compiled_reel.mp4'
    compile_video_segments(segment_paths, audio_path, compiled_video_path)

    return compiled_video_path

# Streamlit UI
st.title("Video to Reels Generator")

# Check if user is logged in
if "username" not in st.session_state:
    st.subheader("Welcome! Please log in or sign up.")
    
    # Option to login or sign up
    option = st.radio("Select Option:", ("Login", "Sign Up"))

    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = auth.check_credentials(username, password)
            if user:
                st.session_state["username"] = username
                st.session_state["user_data"] = user  # Store user data
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid username or password")
    
    elif option == "Sign Up":
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        occupation = st.text_input("Occupation")
        email = st.text_input("Email")
        phone_number = st.text_input("Phone Number")

        if st.button("Sign Up"):
            if all([first_name, last_name, username, password, occupation, email, phone_number]):
                auth.signup(first_name, last_name, username, password, occupation, email, phone_number)
                st.success("Signup successful! Please log in.")
            else:
                st.error("Please fill in all the fields.")
else:
    st.success(f"Welcome, {st.session_state['username']}!")
    
    # Navigation bar
    nav_option = st.sidebar.radio("Navigation", ("Home", "Profile", "Logout"))

    if nav_option == "Home":
        # Home page for video upload
        st.subheader("Upload a video to generate a reel:")
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            video_path = uploaded_file.name
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded file: {uploaded_file.name}")

            if st.button("Generate Reel"):
                with st.spinner('Generating reel...'):
                    compiled_video_path = generate_reel_from_important_segments(video_path)
                st.video(compiled_video_path)
                st.success("Reel generated successfully!")
                st.write(f"Download reel: {compiled_video_path}")

    elif nav_option == "Profile":
        # Profile page for user details
        st.subheader("Profile Information")
        user_data = st.session_state['user_data']
        st.write(f"**First Name:** {user_data[3]}")
        st.write(f"**Last Name:** {user_data[4]}")
        st.write(f"**Username:** {st.session_state['username']}")
        st.write(f"**Occupation:** {user_data[5]}")
        st.write(f"**Email:** {user_data[1]}")
        st.write(f"**Phone Number:** {user_data[7]}")

        profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"])
        if profile_pic is not None:
            # Save the profile picture (handle storage accordingly)
            profile_pic_path = f"profile_pics/{st.session_state['username']}.jpg"
            with open(profile_pic_path, "wb") as f:
                f.write(profile_pic.getbuffer())
            st.success("Profile picture updated!")

    elif nav_option == "Logout":
        st.session_state.clear()  # Clear session
        st.success("Logged out successfully. Redirecting to login page...")
        st.rerun()  # Reload the app to show login/signup page
