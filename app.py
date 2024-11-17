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
from pytube import YouTube
import yt_dlp

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logging.error("OpenAI API key not set in environment variables.")
    st.error("OpenAI API key is required. Please set it in your environment.")
    st.stop()

@st.cache_data
def download_youtube_video(youtube_url):
    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': tempfile.mktemp(suffix='.mp4'),  # Temporary file
            'noplaylist': True,  # Avoid downloading playlist if URL is a playlist
        }

        # Use yt-dlp to download video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_file_path = ydl.prepare_filename(info_dict)
            
            # Reduce file size after downloading
            reduced_video_path = tempfile.mktemp(suffix='_reduced.mp4')
            ffmpeg.input(video_file_path).output(reduced_video_path, vf='scale=640:360', video_bitrate='500k', vcodec='libx264').run()

            # Check if the file was reduced correctly
            if os.path.getsize(reduced_video_path) > 0:
                logging.info(f"Downloaded and reduced video file size: {os.path.getsize(reduced_video_path)} bytes.")
                st.session_state["video_path"] = reduced_video_path
                return reduced_video_path
            else:
                logging.error("Downloaded and reduced video is empty.")
                st.error("Video download or reduction failed.")
                return None
    except Exception as e:
        logging.error(f"Error downloading YouTube video: {e}")
        st.error("Failed to download YouTube video. Please check the URL and try again.")
        return None



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
            {"role": "system", "content": "You are a helpful assistant that understands the genre and motive of a given text and gives the essence of the text."},
            {"role": "user", "content": f"Understand the genre and motive of this text and give its essence: {transcribed_text}"}
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
                {"role": "user", "content": f"Relevant to the genre and motive: '{video_summary}', rate the relevance of: {text} on a scale of 0 to 10. Just give the rating in one digit without any words."}
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

# Save important segments to file with unique timestamped name
def save_segments_to_file(segments, filename_prefix='important_segments'):
    timestamp = int(time.time())  # Get current timestamp
    filename = f"{filename_prefix}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(segments, f, indent=4)
    logging.info(f"Segments saved to {filename}")

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
        audio_inputs = [ ffmpeg.input(audio) for audio in segment_audio_paths]

        # Concatenate video segments with matching audio streams
        video_concat = ffmpeg.concat(*video_inputs, v=1, a=0)
        audio_concat = ffmpeg.concat(*audio_inputs, v=0, a=1)

        # Output the final file with concatenated video and audio streams
        ffmpeg.output(video_concat, audio_concat, output_video_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)
        logging.info(f"Video compiled successfully to {output_video_path}")

    except ffmpeg.Error as e:
        logging.error(f'FFmpeg error: {e.stderr.decode("utf-8") if e.stderr else "Unknown error"}')
        st.error("An error occurred during video processing. Check logs for details.")

def generate_reels_from_important_segments(video_path):
    timestamp = int(time.time())  # Unique timestamp for each session
    audio_path = f'output_audio_{timestamp}.mp3'
    extract_audio(video_path, audio_path)
    transcribed_text, segments = transcribe_audio(audio_path)

    video_summary = get_video_summary(transcribed_text)
    important_segments = analyze_text_importance_and_theme(segments, video_summary)
    important_segments.sort(key=lambda x: x['importance_score'], reverse=True)

    # Split important segments into 3 reels of 30 seconds each
    reels = [[] for _ in range(3)]
    durations = [0] * 3

    for segment in important_segments:
        segment_duration = segment['end_time'] - segment['start_time']
        # Add segment to the reel with the least total duration
        for i in range(3):
            if durations[i] + segment_duration <= 30:
                reels[i].append(segment)
                durations[i] += segment_duration
                break

    # Save important segments to file
    save_segments_to_file({"Reel_1": reels[0], "Reel_2": reels[1], "Reel_3": reels[2]})

    # Generate video and audio for each reel
    reel_video_paths = []
    for reel_index, reel in enumerate(reels):
        segment_video_paths = []
        segment_audio_paths = []

        for i, segment in enumerate(reel):
            start_time = segment['start_time']
            end_time = segment['end_time']
            video_output_path = f'segment_video_{timestamp}_reel{reel_index + 1}_{i + 1}.mp4'
            audio_output_path = f'segment_audio_{timestamp}_reel{reel_index + 1}_{i + 1}.aac'
            extract_video_audio_segment(video_path, start_time, end_time, video_output_path, audio_output_path)
            segment_video_paths.append(video_output_path)
            segment_audio_paths.append(audio_output_path)

        # Compile each reel
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            compiled_video_path = tmp_file.name
        compile_video_segments_with_audio(segment_video_paths, segment_audio_paths, compiled_video_path)
        reel_video_paths.append(compiled_video_path)

    return reel_video_paths

# Streamlit Layout
st.set_page_config(page_title="Video to Reels Generator", layout="wide")

# Landing Page (Only visible if user is not logged in)
if "username" not in st.session_state:
    st.title("Welcome to Video to Reels Summarizer")
    st.image("C://Users//ajay//OneDrive//Desktop//VR//Video-to-Reels_oct_2024//VR_bg.jpg", width=700)  # Add your logo or landing image here
    st.markdown("""
    <style>
        @font-face {
            font-family: 'ZenDots-Regular';
            src: url('fonts/ZenDots-Regular.ttf') format('truetype');
        }

        h3 {
            font-family: 'ZenDots-Regular', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)
    st.markdown(
        """
        <h3>This application transforms your videos into concise, captivating 30-second reels. 
        Log in or sign up to start summarizing your videos!</h3>""", unsafe_allow_html=True
    )
    
    if st.button("Get Started", key="get_started"):
        st.session_state["show_login"] = True
        st.rerun()  # Ensures we show the login/signup form without showing the landing page again.

else:
    st.session_state["show_login"] = False

# Login or Signup Flow (Visible if 'show_login' is True)
if "show_login" in st.session_state and st.session_state["show_login"]:
    st.title("Login or Sign Up")
    st.markdown("### Please select your action:")

    # Show radio buttons for Login and Signup
    option = st.radio("Choose an action:", ["Login", "Sign Up"], key="auth_option", horizontal=True)

    # Login Flow
    if option == "Login":
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        # Login action button (appears after both username and password are filled)
        if username and password:
            login_action = st.button("Login", key="login_action")

            if login_action:
                user = auth.check_credentials(username, password)
                if user:
                    st.session_state["username"] = username
                    st.session_state["user_data"] = user
                    st.success(f"Welcome, {username}!")
                    st.session_state["show_login"] = False  # Hide login form after successful login
                    st.rerun()  # Trigger page refresh after login
                else:
                    st.error("Invalid username or password.")
        else:
            st.warning("Please enter both username and password.")

    # Signup Flow
    elif option == "Sign Up":
        first_name = st.text_input("First Name", key="signup_first_name")
        last_name = st.text_input("Last Name", key="signup_last_name")
        username = st.text_input("Username", key="signup_username")
        password = st.text_input("Password", type="password", key="signup_password")
        occupation = st.text_input("Occupation", key="signup_occupation")
        email = st.text_input("Email", key="signup_email")
        phone_number = st.text_input("Phone Number", key="signup_phone_number")
        profile_picture = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"], key="signup_profile_picture")

        if st.button("Sign Up", key="signup_action"):
            if all([first_name, last_name, username, password, occupation, email, phone_number]) and profile_picture is not None:
                profile_path = f"profiles/{profile_picture.name}"
                with open(profile_path, "wb") as f:
                    f.write(profile_picture.getbuffer())
                # Sign up logic (replace with actual database storage)
                auth.signup(first_name, last_name, username, password, occupation, email, phone_number, profile_path)
                st.success("Signup successful! Please log in.")
            else:
                st.error("Please complete all fields.")
                
# Custom CSS to make buttons look professional
st.markdown("""
    <style>
        .stButton>button {
            background-color: #034078; /* Green background */
            color: white; /* White text */
            border: 1px solid #034078; /* Matching border */
            padding: 15px 32px; /* Padding to make the button bigger */
            font-size: 18px; /* Larger font size */
            border-radius: 5px; /* Rounded corners */
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s; /* Smooth transitions */
        }

        .stButton>button:hover {
            background-color: #001F54; /* Darker green on hover */
            color: white;
            transform: scale(1.05); /* Slight zoom effect */
        }

        .stButton>button:focus {
            outline: none; /* Remove outline when button is focused */
        }

        /* Ensure buttons are displayed horizontally */
        .stButton {
            display: inline-block;
            margin-right: 10px; /* Space between buttons */
        }
    </style>
""", unsafe_allow_html=True)

# Main Application After Login (Visible after successful login)
if "username" in st.session_state:
    # Buttons for Home, Profile, and Logout
    st.title(f"Generate your Moments, {st.session_state["username"]}")

    # Create three columns for horizontal layout
    col1, col2, col3 = st.columns(3)

    # Home Button
    with col1:
        if st.button("Home", key="home_button"):
            st.session_state["current_page"] = "Home"
            st.rerun()  # Refresh the app

    # Profile Button
    with col2:
        if st.button("Profile", key="profile_button"):
            st.session_state["current_page"] = "Profile"
            st.rerun()  # Refresh the app

    # Logout Button
    with col3:
        if st.button("Logout", key="logout_button"):
            st.session_state.pop("username")
            st.session_state.pop("user_data")
            st.session_state.pop("show_login", None)
            st.session_state["current_page"] = "Logout"
            st.rerun()  # Refresh the app

    # Page Content Based on Current Page
    if st.session_state.get("current_page") == "Home":
        st.title("Home - Generate Reels")
        st.markdown("### Choose your input method:")

        # Tabs for file upload and YouTube URL
        tab1, tab2 = st.tabs(["Upload Video File", "YouTube URL"])

        with tab1:
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_video_path = temp_file.name
                st.success("Video uploaded successfully!")
                st.video(temp_video_path)
                if st.button("Generate Reel", key="video"):
                    with st.spinner("Generating reels..."):
                        reel_video_paths = generate_reels_from_important_segments(temp_video_path)
                    if reel_video_paths:
                        for i, reel_path in enumerate(reel_video_paths):
                            st.subheader(f"Reel {i + 1}")
                            st.video(reel_path)
                            with open(reel_path, "rb") as file:
                                st.download_button(f"Download Reel {i + 1}", file.read(), f"reel_{i + 1}.mp4", "video/mp4")
                    else:
                        st.error("Failed to generate reels.")

        with tab2:
            youtube_url = st.text_input("Enter YouTube video URL")
            if st.button("Fetch Video"):
                temp_video_path = download_youtube_video(youtube_url)
                if temp_video_path:
                    st.success("YouTube video downloaded successfully!")
                    st.video(temp_video_path)
                    if st.button("Generate Reel", key="yt_url"):
                        with st.spinner("Generating reels..."):
                            reel_video_paths = generate_reels_from_important_segments(temp_video_path)
                        if reel_video_paths:
                            for i, reel_path in enumerate(reel_video_paths):
                                st.subheader(f"Reel {i + 1}")
                                st.video(reel_path)
                                with open(reel_path, "rb") as file:
                                    st.download_button(f"Download Reel {i + 1}", file.read(), f"reel_{i + 1}.mp4", "video/mp4")
                        else:
                            st.error("Failed to generate reels.")

    elif st.session_state.get("current_page") == "Profile":
        st.title("Your Profile")
        user_data = st.session_state["user_data"]
        if user_data[8]:
            st.image(user_data[8], caption="Your Profile Picture", output_format="PNG", width=150)
        else: st.write("No profile picture uploaded.")
        st.write(f"**First Name:** {user_data[3]}")
        st.write(f"**Last Name:** {user_data[4]}")
        st.write(f"**Occupation:** {user_data[5]}")
        st.write(f"**Username:** {user_data[6]}")
        st.write(f"**Email:** {user_data[1]}")
        st.write(f"**Phone Number:** {user_data[7]}")

    elif st.session_state.get("current_page") == "Logout":
        st.info("Logged out successfully.")
        st.rerun()
