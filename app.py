import streamlit as st 
import auth
import whisper
import ffmpeg
import openai
import os
import json
import logging
import tempfile
import time
from tqdm import tqdm

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
    result = model.transcribe(audio_path, fp16=False)
    return result['text'], result['segments']

# Generating a Summary of the Entire Video
def get_video_summary(transcribed_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes the main topics and essence of a given text."},
            {"role": "user", "content": f"Summarize the following text to capture the main essence: {transcribed_text}"}
        ],
        max_tokens=200,
        temperature=0.5
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Analyzing Text Segments for Theme and Importance using OpenAI API
@st.cache_data
def analyze_text_importance_and_theme(segments, video_summary):
    important_segments = []
    for segment in tqdm(segments, desc="Analyzing segments"):
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing text relevance based on a theme."},
                {"role": "user", "content": f"Based on the theme '{video_summary}', rate the relevance of: {text} on a scale of 0 to 10. Just give the rating in one digit without any words."}
            ],
            max_tokens=100,
            temperature=0.5
        )
        logging.info(f"OpenAI response for segment '{text}': {response}")
        content = response['choices'][0]['message']['content'].strip()
        try:
            importance_score = float(content)
        except (ValueError, IndexError):
            logging.error(f"Unexpected response: {response['choices'][0]['message']['content']}")
            importance_score = 0.0  # Default score if parsing fails

        if importance_score > 5.0:  # Select segments with higher relevance
            important_segments.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'importance_score': importance_score
            })
    return important_segments

# Save important segments to file
# Save important segments to file with unique timestamped name
def save_segments_to_file(segments, filename_prefix='important_segments'):
    timestamp = int(time.time())  # Get current timestamp
    filename = f"{filename_prefix}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(segments, f, indent=4)
    logging.info(f"Segments saved to {filename}")

# Extracting Video Segments Based on Timestamps
def extract_video_segment(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    ffmpeg.input(video_path, ss=start_time, t=duration).output(output_path).run()

# Extracting Video and Audio Segments Based on Timestamps
def extract_video_audio_segment(video_path, start_time, end_time, video_output_path, audio_output_path):
    duration = end_time - start_time
    ffmpeg.input(video_path, ss=start_time, t=duration).output(video_output_path, vcodec='libx264').run()
    ffmpeg.input(video_path, ss=start_time, t=duration).output(audio_output_path, acodec='aac').run()

# Compiling Extracted Segments into a 30-Second Reel with matching audio
def compile_video_segments_with_audio(segment_video_paths, segment_audio_paths, output_video_path):
    try:
        # Create FFmpeg input streams for each segment video and audio
        video_inputs = [ffmpeg.input(video) for video in segment_video_paths]
        audio_inputs = [ffmpeg.input(audio) for audio in segment_audio_paths]

        # Concatenate video segments with matching audio streams
        video_concat = ffmpeg.concat(*video_inputs, v=1, a=0)
        audio_concat = ffmpeg.concat(*audio_inputs, v=0, a=1)

        # Output the final file with concatenated video and audio streams
        ffmpeg.output(video_concat, audio_concat, output_video_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)
        logging.info(f"Video compiled successfully to {output_video_path}")

    except ffmpeg.Error as e:
        logging.error(f'FFmpeg error: {e.stderr.decode("utf-8") if e.stderr else "Unknown error"}')
        st.error("An error occurred during video processing. Check logs for details.")

def generate_reel_from_important_segments(video_path):
    timestamp = int(time.time())  # Unique timestamp for each session
    audio_path = f'output_audio_{timestamp}.mp3'
    extract_audio(video_path, audio_path)
    transcribed_text, segments = transcribe_audio(audio_path)

    video_summary = get_video_summary(transcribed_text)
    important_segments = analyze_text_importance_and_theme(segments, video_summary)
    important_segments.sort(key=lambda x: x['importance_score'], reverse=True)

    selected_segments = []
    total_duration = 0
    for segment in important_segments:
        segment_duration = segment['end_time'] - segment['start_time']
        if total_duration + segment_duration <= 35:
            selected_segments.append(segment)
            total_duration += segment_duration
        else:
            break

    save_segments_to_file(selected_segments)

    segment_video_paths = []
    segment_audio_paths = []
    for i, segment in enumerate(selected_segments):
        start_time = segment['start_time']
        end_time = segment['end_time']
        video_output_path = f'segment_video_{timestamp}_{i + 1}.mp4'
        audio_output_path = f'segment_audio_{timestamp}_{i + 1}.aac'
        extract_video_audio_segment(video_path, start_time, end_time, video_output_path, audio_output_path)
        segment_video_paths.append(video_output_path)
        segment_audio_paths.append(audio_output_path)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        compiled_video_path = tmp_file.name

    compile_video_segments_with_audio(segment_video_paths, segment_audio_paths, compiled_video_path)

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
        profile_picture = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"])

        if st.button("Sign Up"):
            if all([first_name, last_name, username, password, occupation, email, phone_number]) and profile_picture is not None:
                # Save the uploaded profile picture to a directory
                profile_picture_path = f"profiles/{profile_picture.name}"  # Specify your upload directory
                with open(profile_picture_path, "wb") as f:
                    f.write(profile_picture.getbuffer())
            
                auth.signup(first_name, last_name, username, password, occupation, email, phone_number, profile_picture_path)
                st.success("Signup successful! Please log in.")
            else:
                st.error("Please fill in all the fields.")

else:
    st.success(f"Welcome, {st.session_state['username']}!")
    
    # Navigation bar
    nav_option = st.sidebar.radio("Navigation", ("Home", "Profile", "Logout"))

    if nav_option == "Home":
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
                if os.path.exists(compiled_video_path):
                    st.video(compiled_video_path)
                    st.success("Reel generated successfully!")
                else:
                    st.error("Reel generation failed: Video file not found.")

    elif nav_option == "Profile":
        st.subheader("Profile Information")
        user_data = st.session_state['user_data']
    
        # Displaying Profile Picture
        if user_data[8]:  # Assuming profile picture URL is the 9th column
            st.image(user_data[8], width=150)  # Adjust the width as needed
        
        st.write(f"**First Name:** {user_data[3]}")
        st.write(f"**Last Name:** {user_data[4]}")
        st.write(f"**Occupation:** {user_data[5]}")
        st.write(f"**Username:** {user_data[6]}")
        st.write(f"**Email:** {user_data[1]}")
        st.write(f"**Phone Number:** {user_data[7]}")


    elif nav_option == "Logout":
        st.session_state.pop("username", None)
        st.session_state.pop("user_data", None)
        st.info("You have been logged out.")
        st.rerun()