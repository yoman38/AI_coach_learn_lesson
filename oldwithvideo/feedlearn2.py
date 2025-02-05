import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings, AudioProcessorBase
import numpy as np
import sqlite3
import bcrypt
import tempfile
import time
import os
from google.cloud import vision
from google.cloud import speech
from google.cloud import texttospeech
import wave
import openai
from openai import OpenAI
import requests
import json

# Set environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\\MesProjets\\gaze_speech_loop\\API\\mapsrandom-427018-03efcd40b0d9.json"

# Initialize Google Cloud Clients
vision_client = vision.ImageAnnotatorClient()
speech_client = speech.SpeechClient()

# OpenAI API Key
openai.api_key = "api"

# Database Setup
def init_db():
    conn = sqlite3.connect("test_users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL
        )
    ''')
    c.execute("SELECT * FROM users WHERE email = ?", ("test",))
    if not c.fetchone():
        hashed_password = bcrypt.hashpw("test".encode(), bcrypt.gensalt()).decode()
        c.execute("INSERT INTO users (email, password, name) VALUES (?, ?, ?)", ("test", hashed_password, "Test User"))
        conn.commit()
    conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect("test_users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode(), user[2].encode()):
        return {"email": user[1], "name": user[3]}
    return None

def create_user(email, password, name):
    conn = sqlite3.connect("test_users.db")
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        c.execute("INSERT INTO users (email, password, name) VALUES (?, ?, ?)", (email, hashed_password, name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def transcribe_audio_with_status(audio_path, language_code):
    st.info("⏳ Transcription in progress... Please wait.")
    try:
        with wave.open(audio_path, "rb") as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            audio_content = audio_file.readframes(frames)

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=rate,
            language_code=language_code,
        )
        response = speech_client.recognize(config=config, audio=audio)
        
        st.success("✅ Transcription complete!")
        return "\n".join(result.alternatives[0].transcript for result in response.results)
    except Exception as e:
        st.error(f"⚠️ Transcription failed: {str(e)}")
        return None


# Initialize the database
init_db()

# Authentication and Login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

if not st.session_state.authenticated:
    st.title("Login or Sign Up")
    auth_mode = st.radio("Select Mode", options=["Login", "Sign Up"], index=0)

    if auth_mode == "Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            user = authenticate_user(email, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.success(f"Welcome back, {user['name']}!")
            else:
                st.error("Invalid email or password.")

    elif auth_mode == "Sign Up":
        with st.form("signup_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign Up")

        if submit:
            success = create_user(email, password, name)
            if success:
                st.success("Account created successfully! You can now log in.")
            else:
                st.error("Email already exists. Please try again.")

    st.stop()

# Logout Option
if st.session_state.authenticated:
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(authenticated=False, user=None))

# App Description
st.title("Interactive App with OCR, Timer, and ChatGPT Integration")
st.markdown(
    """
    **How to use this app:**
    1. **Left Column:** Upload an image or paste text. OCR will extract text from images.
    2. **Middle Column:** Start a 1-minute timer, then record your voice. Transcription will follow.
    3. **Right Column:** ChatGPT analyzes and provides insights based on the left column input.
    """
)

# Three-column layout
left_col, middle_col, right_col = st.columns(3)

# --------------------
# Left Column: Input Zone
# --------------------
with left_col:
    st.header("Input Zone")
    image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    text_input = st.text_area("Paste text here:")
    extracted_text = ""

    if image:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image.read())
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as img_file:
            content = img_file.read()
            image = vision.Image(content=content)
            response = vision_client.text_detection(image=image)
            extracted_text = response.text_annotations[0].description if response.text_annotations else ""

        st.success("✅ OCR successful!")
        st.write("Extracted Text:")
        st.write(extracted_text)

    if text_input:
        st.success("✅ Text pasted successfully!")
        st.write("Your Input Text:")
        st.write(text_input)

# --------------------
# Middle Column: Timer and Voice Recording
# --------------------
with middle_col:
    st.header("Timer & Audio Upload")

    # Persistent state for language selection
    if "lang" not in st.session_state:
        st.session_state.lang = "en-US"  # Default language

    # Language selection (independent of the rest of the logic)
    st.session_state.lang = st.selectbox(
        "Select Language",
        options=["en-US", "fr-FR", "es-ES"],
        index=["en-US", "fr-FR", "es-ES"].index(st.session_state.lang)
    )

    # Timer functionality (if still wanted)
    if st.button("Start Timer"):
        st.info("Timer started for 1 minute...")
        with st.empty():
            for i in range(10, 0, -1):  # Set to 10 seconds for faster testing
                st.write(f"Time remaining: {i} seconds")
                time.sleep(1)

        st.success("Timer complete! Upload your audio file.")

    # Audio file upload (Drag and Drop functionality)
    st.write("Upload an audio file for transcription.")
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

    if audio_file:
        st.success("Audio file uploaded successfully!")
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name

        # Transcribe the audio file using Google Speech-to-Text
        transcription = transcribe_audio_with_status(audio_path, st.session_state.lang)
        st.success("Transcription complete!")
        st.write("Transcript:")
        st.write(transcription)

# --------------------
# Right Column: ChatGPT Output
# --------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with right_col:
    st.header("ChatGPT Output")
    
    # Language Mapping
    language_mapping = {
        "en-US": "English",
        "fr-FR": "French",
        "es-ES": "Spanish",
    }

    # Determine the selected language
    selected_language = language_mapping.get(st.session_state.lang, "English")

    # Determine the source for lesson content
    if extracted_text.strip():
        lesson_to_know = extracted_text.strip()
        st.write("✅ Using extracted text from the image as the lesson content.")
    elif text_input.strip():
        lesson_to_know = text_input.strip()
        st.write("✅ Using user-entered text as the lesson content.")
    else:
        lesson_to_know = None
        st.warning("⚠️ No lesson content available. Please upload an image or enter text.")

    # Initialize session state for transcription
    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "transcription_status" not in st.session_state:
        st.session_state.transcription_status = "idle"  # idle, in_progress, completed

    # Trigger transcription only when audio is uploaded
    if audio_file and st.session_state.transcription_status == "idle":
        st.session_state.transcription_status = "in_progress"
        with st.spinner("Transcribing audio..."):
            st.session_state.transcription = transcribe_audio_with_status(audio_path, st.session_state.lang)
        st.session_state.transcription_status = "completed"

    # Process ChatGPT once transcription is ready
    if lesson_to_know and st.session_state.transcription_status == "completed" and st.session_state.transcription:
        user_transcription = st.session_state.transcription.strip()

        # Construct the ChatGPT input
        chat_input = (
            f"You're a language teacher who provides friendly, oral feedback to students. "
            f"**Respond in {selected_language} writing in plain text** and use a conversational tone, as if you're talking directly to the student.\n\n"
            f"Analyze and compare the following information:\n\n"
            f"1. User Transcription:\n{user_transcription}\n\n"
            f"2. Lesson to Know:\n{lesson_to_know}\n\n"
            "Provide feedback that:\n"
            "1. Highlights what the student got right.\n"
            "2. Highlights what the student got wrong and correct it.\n"
            "3. Points out additional details they included that were not in the lesson to know, if any. \n"
            "4. Explains any important information they missed, without missing any (even if it is long) and offers tips for improvement."
            "5. Explains any important information they missed, without missing any (even if it is long) and offers tips for improvement."
            "Do **not** use chapters or bold text or * in your output. Write like a young woman speak orally."
        )

        # Call OpenAI API to process the comparison
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": chat_input}],
            max_tokens=4000
        )
        
        # Ensure `response` is valid before proceeding
        if response.choices and response.choices[0].message.content:
            chat_response = response.choices[0].message.content

            # Display the response
            st.success("ChatGPT Response:")
            st.write(chat_response)


            # --------------------
            # TTS Integration
            # --------------------
            # D-ID API key and configuration
            d_id_api_key = "Ymd1aXlvbTM4QGdtYWlsLmNvbQ:cJ_xitVmX1ZjYhaGu-w_2"
            image_url = "s3://d-id-images-prod/google-oauth2|102030034095188114506/img_44udWEp9oqZQvIGf1bgYX/webcam.jpg"  # Replace with the actual URL of your uploaded image

            # Function to create D-ID video
            def create_did_video(message):
                data = {
                    "source_url": image_url,
                    "script": {
                        "type": "text",
                        "input": message,
                        "provider": {
                            "type": "microsoft",
                            "voice_id": "Fr-FR-RemyMultilingualNeural"
                        }
                    }
                }

                headers = {
                    "accept": "application/json",
                    "authorization": f"Basic {d_id_api_key}",
                    "Content-Type": "application/json"
                }

                # Send POST request to create talk
                response = requests.post("https://api.d-id.com/talks", headers=headers, data=json.dumps(data))

                if response.status_code == 201:
                    response_data = response.json()
                    talk_id = response_data.get("id")
                    status_url = f"https://api.d-id.com/talks/{talk_id}"

                    # Polling the status of the video
                    while True:
                        status_response = requests.get(status_url, headers=headers)
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            status = status_data.get("status")
                            if status == "done":
                                video_url = status_data.get("result_url")
                                return video_url
                            else:
                                time.sleep(5)
                        else:
                            print(f"Error checking video status. Status Code: {status_response.status_code}")
                            break
                else:
                    print(f"Error creating talk. Status Code: {response.status_code}")
                    print("Response:", response.text)
                    return None

            # Replace TTS with D-ID video in Streamlit app
            if "authenticated" in st.session_state and st.session_state.authenticated:
                # ChatGPT-generated message from transcription (example)
                chat_response = chat_response

                # Create D-ID video from the chat response
                video_url = create_did_video(chat_response)

                if video_url:
                    st.success("Video is ready! Watch it below.")
                    st.video(video_url)  # Display the video in the Streamlit UI
                else:
                    st.error("There was an error creating the video.")