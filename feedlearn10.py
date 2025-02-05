import streamlit as st
import numpy as np
import tempfile
import time
import os
import re  # For parsing resume formatting and parts
from google.cloud import vision
from google.cloud import speech
from google.cloud import storage  # For uploading long audio files to GCS (if needed)
import wave
import openai
from openai import OpenAIError
import requests
import json
import av  # Needed for audio frame conversion
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings, AudioProcessorBase
import PyPDF2  # For PDF text extraction

# ----------------------------
# Helper: Convert language codes to ISO-639-1
# ----------------------------
def iso_lang(language_code):
    # Example: "en-US" becomes "en", "fr-FR" becomes "fr", "es-ES" becomes "es"
    return language_code.split("-")[0].lower()

# ----------------------------
# Custom Audio Processor Class
# ----------------------------
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.sample_rate = None  # Set from the first received frame

    def recv(self, frame):
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
        audio_array = frame.to_ndarray()
        self.frames.append(audio_array)
        return frame

    def get_audio_data(self):
        if self.frames:
            return np.concatenate(self.frames, axis=0)
        return None

# ----------------------------
# Environment & Client Setup
# ----------------------------
# Load credentials from Streamlit Secrets
gcs_credentials = st.secrets["google_cloud"]["credentials"]

# Check if credentials is a string or already a dict
if isinstance(gcs_credentials, str):
    creds = json.loads(gcs_credentials)
else:
    creds = gcs_credentials

# Save to a temporary file
with open("google_credentials.json", "w") as f:
    json.dump(creds, f)

# Set the environment variable for Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

# Initialize the Google Cloud clients
vision_client = vision.ImageAnnotatorClient()
speech_client = speech.SpeechClient()
storage_client = storage.Client()

# Set the name of your GCS bucket here (or load it from secrets).
GCS_BUCKET_NAME = st.secrets["secrets"]["gcs_bucket_name"]

# Set OpenAI API key securely from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
# For convenience, we alias the OpenAI client:

# Load OpenAI API key securely from Streamlit Secrets
API_KEY = st.secrets["openai"]["api_key"]

client = openai.Client(api_key=API_KEY)

# ----------------------------
# Helper Function: Transcribe Audio using OpenAI Whisper
# ----------------------------
def transcribe_audio_with_status(audio_path, language_code):
    st.info("⏳ Transcription in progress... Please wait.")
    try:
        with open(audio_path, "rb") as audio_file:
            # Use the ISO-639-1 language code for Whisper.
            lang = iso_lang(language_code)
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=lang,  # Now a two-letter code (e.g. "fr")
                response_format="text"
            )
        # Check if the returned object is a string or has a 'text' attribute.
        transcript = transcription if isinstance(transcription, str) else transcription.text
        st.success("✅ Transcription complete!")
        return transcript
    except Exception as e:
        st.error(f"⚠️ Transcription failed: {str(e)}")
        return None

# ----------------------------
# Step 4 Helper Functions: Exercise Generation and Answer Checking
# ----------------------------
def generate_exercise(topic, selected_language):
    """
    Generate an exercise about the given topic.
    The exercise will consist of 7 multiple choice questions, each formatted as:
    
    1.
    Question: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]

    Ensure that the formatting is consistent and clear.
    """
    prompt = f"""
You are a helpful assistant that creates educational exercises.
Write an exercise using informations from: {topic}. Respond in {selected_language}.
The exercise should consist of 10 multiple choice questions, with 1 correct answer.
Each question should follow this format:

1.
Question: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

Ensure that the formatting is consistent and clear. Do not provide answers, only provide questions following the format. Make sure to have only 1 valid answer.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating the exercise: {e}")
        return ""

def check_answers(topic, exercise, answers, selected_language):
    """
    Check the user's answers against the exercise.
    'answers' should be formatted as "1A, 2C, ..." (question number with chosen option letter).
    """
    prompt = f"""
You are a teacher.
Exercise:
{exercise}

User's Answers:
{answers}

For each question, check whether the answer is correct and provide an explanation for why it is correct or incorrect. Respond in {selected_language}
If needed, use informations from: {topic}
/n
Present the results in the following format:

1. [Correct/Incorrect]
Explanation: ...

2. [Correct/Incorrect]
Explanation: ...

...
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a teacher."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while checking the answers: {e}")
        return ""

# ----------------------------
# Step 5 Helper Functions: Resume Generation and Parsing
# ----------------------------
def generate_resume(lesson, selected_language):
    """
    Use ChatGPT to generate a concise resume summary (memo sheet) based solely on the provided lesson content.
    IMPORTANT: Generate the summary in a structured format with multiple parts.
    Each part should be structured as follows (use plain text inside the tags and mark important words as needed):

    <part>
    <title> Title text with importance markers </title>
    <content> Content text with importance markers </content>
    <example> Example text with importance markers </example>
    </part>

    Separate each part by a newline. Do not include any additional HTML tags.
    """
    prompt = f"""
You are an experienced teacher tasked with creating a concise resume summary memo sheet.
Based solely on the following lesson content, create a summary that highlights the key notions. Respond in {selected_language}.
Make sure to use informations from the lesson content, and think twice if your examples are relevant or not. 
IMPORTANT: Structure your summary into multiple parts. For each notion, follow this exact format:

<part>
<title> Title text (you may mark extremely important words with *** , moderately with ** , and slightly with * ) </title>
<content> Content text (you may mark important words as above) </content>
<example> Example text (you may mark important words as above) </example>
</part>

Lesson Content:
{lesson}

Provide the summary in plain text with no additional HTML tags.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an experienced teacher."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating the resume: {e}")
        return ""

def parse_resume_formatting(text):
    """
    Parse the resume text for special markers and replace them with HTML spans:
      - ***...*** -> highlight (e.g., yellow background)
      - **...** -> bold text
      - *...* -> underlined text
    """
    # Replace triple asterisks first for the highest importance.
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<span class="highlight">\1</span>', text)
    # Then replace double asterisks for moderately important words.
    text = re.sub(r'\*\*(.+?)\*\*', r'<span class="bold">\1</span>', text)
    # Finally, replace single asterisks for slightly important words.
    text = re.sub(r'\*(.+?)\*', r'<span class="underline">\1</span>', text)
    return text

def parse_resume_parts(text):
    """
    Parse the structured resume text into parts and convert each part into HTML.
    Each part is expected to be wrapped in <part> ... </part> tags and to include:
       <title> ... </title>
       <content> ... </content>
       <example> ... </example>
    The markers for importance are processed via parse_resume_formatting.
    """
    parts_html = ""
    # Find all parts defined by <part> ... </part>
    parts = re.findall(r'<part>(.*?)</part>', text, flags=re.DOTALL)
    if not parts:
        # If no <part> markers found, treat entire text as one part.
        parts = [text]
    for part in parts:
        # Extract the title, content, and example
        title_match = re.search(r'<title>(.*?)</title>', part, flags=re.DOTALL)
        content_match = re.search(r'<content>(.*?)</content>', part, flags=re.DOTALL)
        example_match = re.search(r'<example>(.*?)</example>', part, flags=re.DOTALL)
        title = parse_resume_formatting(title_match.group(1).strip()) if title_match else ""
        content = parse_resume_formatting(content_match.group(1).strip()) if content_match else ""
        example = parse_resume_formatting(example_match.group(1).strip()) if example_match else ""
        part_html = "<div class='resume-part'>"
        if title:
            part_html += f"<h2 class='resume-title'>{title}</h2>"
        if content:
            part_html += f"<p class='resume-content'>{content}</p>"
        if example:
            part_html += f"<p class='resume-example'><strong>Example:</strong> <em>{example}</em></p>"
        part_html += "</div>"
        parts_html += part_html
    return parts_html

# ----------------------------
# Main App UI
# ----------------------------
st.title("Interactive App: OCR, Transcription, ChatGPT Feedback, Exercise Generation & Resume")
st.markdown(
    """
This app guides you through a five‑step process:

**Step 1:** Provide your lesson content by uploading an image/PDF (OCR will extract text) or by pasting text.
  
**Step 2:** Start a 10‑second timer and then upload or record an audio file for transcription.

**Step 3:** Get ChatGPT feedback comparing your transcription with your lesson content along with text‑to‑speech of the feedback.

**Step 4:** Generate exercises (multiple choice questions) based on a topic.

**Step 5:** Generate a resume summary (memo sheet) based solely on your lesson content.
"""
)

# ----------------------------
# Step 1: Lesson Content Input
# ----------------------------
st.header("Step 1: Lesson Content Input")
st.markdown("Upload an image (jpg, jpeg, png), a PDF file, or paste text below.")

# Allow image/PDF upload in separate uploaders for clarity.
image_file = st.file_uploader("Upload an image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])
pdf_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
text_input = st.text_area("Or paste your text here:")

# Variables to store extracted text from image or PDF.
extracted_text = ""
extracted_pdf_text = ""

if image_file:
    st.info("Processing image for OCR...")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(image_file.read())
        temp_file_path = temp_file.name
    with open(temp_file_path, "rb") as img_file:
        content = img_file.read()
        image_for_ocr = vision.Image(content=content)
        response = vision_client.text_detection(image=image_for_ocr)
        extracted_text = response.text_annotations[0].description if response.text_annotations else ""
    st.success("✅ OCR successful!")
    st.write("**Extracted Text from Image:**")
    st.write(extracted_text)

if pdf_file:
    st.info("Processing PDF file for text extraction...")
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_pdf_text += page.extract_text() + "\n"
        st.success("✅ PDF text extracted!")
        st.write("**Extracted Text from PDF:**")
        st.write(extracted_pdf_text)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")

# Button to send/confirm the text to be used as lesson content.
if st.button("Send Text"):
    # Priority: PDF > Image > Pasted text.
    if extracted_pdf_text.strip():
        lesson_content = extracted_pdf_text.strip()
        st.info("Using extracted PDF text as lesson content.")
    elif extracted_text.strip():
        lesson_content = extracted_text.strip()
        st.info("Using extracted image text as lesson content.")
    elif text_input.strip():
        lesson_content = text_input.strip()
        st.info("Using your pasted text as lesson content.")
    else:
        lesson_content = None
        st.warning("No lesson content provided.")
    st.session_state.lesson_content = lesson_content  # Save in session state for later use.

# If lesson content is already set, display it.
if st.session_state.get("lesson_content"):
    st.markdown("### Lesson Content to Use:")
    st.write(st.session_state.lesson_content)

# ----------------------------
# Step 2: Audio Input and Timer
# ----------------------------
st.header("Step 2: Audio Input and Timer")
st.markdown("Click the button to start a 10‑second timer (for testing), then upload or record an audio file for transcription.")

if "transcription_language" not in st.session_state:
    st.session_state.transcription_language = "en-US"
st.session_state.transcription_language = st.selectbox(
    "Select transcription language:",
    options=["en-US", "fr-FR", "es-ES"],
    index=["en-US", "fr-FR", "es-ES"].index(st.session_state.transcription_language)
)

if st.button("Start Timer (10 seconds)"):
    st.info("Timer started for 10 seconds...")
    timer_placeholder = st.empty()
    for i in range(10, 0, -1):
        timer_placeholder.write(f"Time remaining: {i} seconds")
        time.sleep(1)
    timer_placeholder.write("Timer complete!")

audio_method = st.radio("Choose audio input method:", options=["Upload Audio File", "Record via Microphone"])
if audio_method == "Upload Audio File":
    st.write("Upload an audio file for transcription:")
    audio_file = st.file_uploader("Choose an audio file:", type=["wav", "mp3", "flac"], key="audio_file")
    if audio_file:
        st.success("Audio file uploaded!")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name
        transcription = transcribe_audio_with_status(audio_path, st.session_state.transcription_language)
        st.session_state.transcription = transcription
        st.session_state.transcription_status = "completed"
        st.write("**Transcript:**")
        st.write(transcription)
else:
    st.write("Record audio via microphone:")
    recorded_audio = st.audio_input("Record a voice message")
    if recorded_audio:
        st.success("Recording captured!")
        st.audio(recorded_audio)
        # Save recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(recorded_audio.read())
            audio_path = temp_audio.name
        transcription = transcribe_audio_with_status(audio_path, st.session_state.transcription_language)
        st.session_state.transcription = transcription
        st.session_state.transcription_status = "completed"
        st.write("**Transcript:**")
        st.write(transcription)

# ----------------------------
# Step 3: ChatGPT Feedback and Text-to-Speech using OpenAI TTS
# ----------------------------
st.header("Step 3: ChatGPT Feedback")
st.markdown("If both lesson content and transcription are available, get ChatGPT feedback comparing them along with text‑to‑speech of the feedback.")

if st.session_state.get("lesson_content") and st.session_state.get("transcription_status") == "completed" and st.session_state.get("transcription"):
    user_transcription = st.session_state.transcription.strip()
    # Use the transcription language selected by the user.
    selected_language = st.session_state.transcription_language
    # Alias the lesson content as the lesson to know.
    lesson_to_know = st.session_state.lesson_content
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": chat_input}],
        max_tokens=4000,
        temperature=0.7
    )
    if response.choices and response.choices[0].message.content:
        chat_feedback = response.choices[0].message.content
        st.success("ChatGPT Feedback:")
        st.write(chat_feedback)
        st.session_state.chat_feedback = chat_feedback  # Save feedback if needed later

        # --- Text-to-Speech of ChatGPT Feedback using OpenAI TTS ---
        try:
            tts_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",  # Change voice as needed
                input=chat_feedback
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                tts_response.stream_to_file(tts_file.name)
                tts_audio_path = tts_file.name
            st.markdown("### ChatGPT Feedback (Audio)")
            st.audio(tts_audio_path, format="audio/mp3")
        except Exception as e:
            st.error(f"Text-to-Speech conversion failed: {e}")
else:
    st.info("Waiting for both lesson content and transcription for ChatGPT feedback...")

# ----------------------------
# Step 4: Exercise Generation
# ----------------------------
st.header("Step 4: Exercise Generation")
st.markdown(
    "Generate exercises (multiple choice questions) based on a topic. " +
    "If lesson content is available from Step 1, it is used by default; otherwise, you can enter a topic manually."
)
default_topic = st.session_state.get("lesson_content") if st.session_state.get("lesson_content") else ""
topic_for_exercise = st.text_area("Enter topic for exercise generation:", value=default_topic)

if st.button("Generate Exercise"):
    with st.spinner("Generating exercise..."):
        exercise_text = generate_exercise(topic_for_exercise, st.session_state.transcription_language)
    if exercise_text:
        st.session_state.exercise = exercise_text
        st.success("Exercise generated successfully!")
        st.markdown("### Generated Exercise")
        st.write(exercise_text)

# If an exercise has been generated, display an answer section
if "exercise" in st.session_state:
    st.subheader("Answer Section")
    user_answers = {}
    with st.form("answer_form"):
        # Iterate through questions and collect answers.
        questions = st.session_state.exercise.split('\n\n')
        for q in questions:
            if not q.strip():
                continue
            lines = q.strip().split('\n')
            if len(lines) < 6:
                st.warning("Some questions might not be formatted correctly. Please regenerate the exercise.")
                continue
            try:
                question_number = lines[0].split('.')[0].strip()
                question_text = lines[1].strip()
                options = [line.strip() for line in lines[2:6]]
                st.markdown(f"**{question_number}. {question_text}**")
                option_texts = []
                for opt in options:
                    if ')' in opt:
                        option_text = opt.split(')', 1)[1].strip()
                    else:
                        option_text = opt
                    option_texts.append(option_text)
                selected_option = st.radio(
                    f"Select your answer for Question {question_number}:",
                    options=option_texts,
                    key=f"q_{question_number}"
                )
                if selected_option:
                    idx = option_texts.index(selected_option)
                    letter = ['A', 'B', 'C', 'D'][idx]
                    user_answers[question_number] = letter
            except Exception as e:
                st.warning(f"Error processing a question: {e}")
        # Submit button for the form.
        submitted = st.form_submit_button("Check Answers")
    
    if submitted:
        st.session_state.user_answers = user_answers
        if st.session_state.exercise and st.session_state.user_answers:
            sorted_answers = sorted(
                st.session_state.user_answers.items(),
                key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')
            )
            formatted_answers = ", ".join([f"{num}{ans}" for num, ans in sorted_answers])
            with st.spinner("Checking answers..."):
                correction = check_answers(
                    st.session_state.lesson_content,
                    st.session_state.exercise,
                    formatted_answers,
                    st.session_state.transcription_language
                )
            if correction:
                st.markdown("### Correction and Explanation")
                st.write(correction)
        else:
            st.warning("Please generate an exercise and answer the questions before checking.")

# ----------------------------
# Step 5: Resume Generation
# ----------------------------
st.header("Step 5: Resume Generation")
st.markdown(
    "Generate a resume summary (memo sheet) using ChatGPT based solely on your lesson content. " +
    "The resume will be structured into several parts. Each part will have a title, content, and example. " +
    "Important words will be marked using *** (extremely), ** (moderately), or * (slightly) and then styled accordingly."
)

if st.button("Generate Resume"):
    if not st.session_state.get("lesson_content"):
        st.warning("Please provide lesson content in Step 1.")
    else:
        lesson = st.session_state.lesson_content
        with st.spinner("Generating resume summary..."):
            resume_text = generate_resume(lesson, st.session_state.transcription_language)
        if resume_text:
            # Parse the structured resume text into HTML parts.
            parsed_resume_html = parse_resume_parts(resume_text)
            # Create an HTML file with the provided styling and the parsed resume content.
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memo Sheets</title>
    <link href="https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Patrick Hand', cursive;
            background-color: #fffde7;
            margin: 0;
            padding: 20px;
            text-align: center;
        }}

        .memo-container {{
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            border: 2px solid #f9d71c;
        }}

        h1 {{
            font-size: 2.5em;
            color: #4caf50;
            text-decoration: underline;
            background: linear-gradient(to right, #ffeb3b, #ff5722);
            -webkit-background-clip: text;
            color: transparent;
        }}

        /* Styles for parsed resume formatting */
        .highlight {{
            background-color: yellow;
            font-weight: bold;
            padding: 2px 5px;
            border-radius: 4px;
        }}

        .bold {{
            font-weight: bold;
        }}

        .underline {{
            text-decoration: underline;
        }}

        /* Styles for resume parts */
        .resume-part {{
            margin-bottom: 20px;
            padding: 10px;
            border-bottom: 1px dashed #ccc;
            text-align: left;
        }}

        .resume-title {{
            font-size: 1.8em;
            color: #ff5722;
            text-decoration: underline;
            margin-bottom: 5px;
        }}

        .resume-content {{
            font-size: 1.2em;
            margin-bottom: 5px;
        }}

        .resume-example {{
            font-size: 1.1em;
            background-color: #e3f2fd;
            padding: 8px;
            border-left: 4px solid #2196f3;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="memo-container">
        <h1>Memo Sheet</h1>
        {parsed_resume_html}
    </div>
</body>
</html>"""
            st.download_button("Download Resume as HTML", html_content, file_name="resume.html", mime="text/html")
            st.success("Resume generated successfully!")
        else:
            st.error("Failed to generate resume.")
