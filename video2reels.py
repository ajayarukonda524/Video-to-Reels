import whisper
import ffmpeg
from textblob import TextBlob
import os
import streamlit as st

# Step 1: Extract Audio from Video using FFmpeg
def extract_audio(video_path, output_audio_path):
    try:
        ffmpeg.input(video_path).output(output_audio_path).run()
        print(f"Audio extracted successfully to {output_audio_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")

# Step 2: Transcribe Audio to Text using OpenAI Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text'], result['segments']

# Step 3: Analyze Text Segments for Importance
def analyze_text_importance(segments):
    important_segments = []
    buffer_time = 0.5  # Buffer to adjust start/end times for smoother cuts

    for segment in segments:
        text = segment['text']
        start_time = max(0, segment['start'] - buffer_time)  # Adding buffer
        end_time = segment['end'] + buffer_time

        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
        word_count = len(text.split())

        # Score the segment based on sentiment and length
        importance_score = sentiment_score * word_count

        # Ensure segment ends with punctuation to complete the sentence
        if text and text[-1] in [".", "!", "?"] and importance_score > 1.0:  # Adjust threshold
            important_segments.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'importance_score': importance_score
            })

    return important_segments

# Step 4: Extract Video Segments Based on Timestamps
def extract_video_segment(video_path, start_time, end_time, output_path):
    try:
        duration = end_time - start_time
        ffmpeg.input(video_path, ss=start_time, t=duration).output(output_path).run()
        print(f"Segment extracted: {output_path}")
    except Exception as e:
        print(f"Error extracting video segment: {e}")

# Step 5: Compile Extracted Segments into 30-Second Reel
def compile_video_segments(segment_paths, output_video_path):
    with open('file_list.txt', 'w') as f:
        for segment in segment_paths:
            f.write(f"file '{segment}'\n")

    try:
        ffmpeg.input('file_list.txt', format='concat', safe=0).output(output_video_path, c='copy').run()
        print(f"Compiled reel created: {output_video_path}")
    except Exception as e:
        print(f"Error compiling videos: {e}")

# Step 6: Save timestamps to a text file
def save_timestamps_to_file(segments, output_file):
    with open(output_file, 'w') as f:
        for segment in segments:
            f.write(f"Start: {segment['start_time']:.2f}, End: {segment['end_time']:.2f}\n")
    print(f"Timestamps saved to {output_file}")

# Full Process: Text Analysis, Timestamp Mapping, Video Extraction, Compilation
def generate_reel_from_important_segments(video_path, audio_path, compiled_video_path, top_n=5):
    # Step 1: Extract audio from video
    extract_audio(video_path, audio_path)

    # Step 2: Transcribe audio and get segment data
    _, segments = transcribe_audio(audio_path)
    print("Transcription and timestamp extraction completed.")

    # Step 3: Analyze text for important segments
    important_segments = analyze_text_importance(segments)

    # Sort by importance score and select top N segments
    important_segments.sort(key=lambda x: x['importance_score'], reverse=True)
    top_segments = important_segments[:top_n]

    # Step 4: Extract corresponding video segments
    segment_paths = []
    for i, segment in enumerate(top_segments):
        start_time = segment['start_time']
        end_time = segment['end_time']
        output_path = f'segment_{i + 1}.mp4'
        extract_video_segment(video_path, start_time, end_time, output_path)
        segment_paths.append(output_path)

    # Step 5: Compile extracted segments into a single reel (max 30 seconds)
    compile_video_segments(segment_paths, compiled_video_path)

    # Save timestamps to a text file
    save_timestamps_to_file(top_segments, "timestamps.txt")  # Save timestamps here

    return top_segments, segment_paths  # Return important segments and segment paths

# Streamlit Interface
def main():
    st.title("Video to Reel Summarizer")

    # Video upload
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file is not None:
        video_path = f"uploaded_video.mp4"
        with open(video_path, mode="wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")

        # Process the video
        if st.button("Generate Reel"):
            audio_path = "extracted_audio.wav"
            compiled_video_path = "compiled_reel.mp4"
            timestamps_file_path = "timestamps.txt"

            st.info("Processing the video...")

            # Generate reel from important segments
            important_segments, segment_paths = generate_reel_from_important_segments(video_path, audio_path, compiled_video_path)

            st.success(f"Reel generated and timestamps saved to {timestamps_file_path}!")

            # Display download link for compiled reel
            with open(compiled_video_path, "rb") as file:
                st.download_button(label="Download Reel", data=file, file_name="compiled_reel.mp4")

            # Provide download link for the timestamps file
            with open(timestamps_file_path, "rb") as file:
                st.download_button(label="Download Timestamps", data=file, file_name="timestamps.txt")

            # Delete segment files after compilation
            for segment_path in segment_paths:
                if os.path.exists(segment_path):
                    os.remove(segment_path)

if __name__ == '__main__':
    main()