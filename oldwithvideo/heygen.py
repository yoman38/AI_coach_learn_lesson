import requests

# Replace with your D-ID API key
API_KEY = "api"

# URL for the D-ID API endpoint to generate the video
url = "https://api.d-id.com/talk"

# Headers for the request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Define the payload with the necessary data
payload = {
    "audio_url": "",  # Optionally, you can upload a pre-recorded audio URL
    "text": "Salut ! Je suis vraiment impressionnée par tes explications sur la conduction, la convection et le rayonnement.",  # The text to speak
    "voice": "en_us_male",  # Voice ID (e.g., "en_us_male" for a male English voice)
    "avatar_url": "https://t3.ftcdn.net/jpg/04/47/31/38/360_F_447313844_CehcFIev9SGc3RrhZyP807nOc71nh6Of.jpg",  # URL to the image of the avatar
    "language": "fr"  # Language code (in this case, French)
}

# Send the POST request to the D-ID API
response = requests.post(url, json=payload, headers=headers)

# Check the response status and get the video URL
if response.status_code == 200:
    response_data = response.json()
    video_url = response_data.get("video_url")
    print(f"Video created successfully! You can download it from: {video_url}")
else:
    print(f"Error creating video. Status Code: {response.status_code}")
    print("Response:", response.text)
