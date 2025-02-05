import requests

# Your D-ID API key (ensure this is set correctly)
d_id_api_key = "api"

# Prepare the image file for upload
files = {"image": ("webcam.jpg", open("webcam.jpg", "rb"), "image/jpeg")}

# Define the URL and headers
url = "https://api.d-id.com/images"
headers = {
    "accept": "application/json",
    "authorization": f"Basic {d_id_api_key}"
}

# Send the POST request
response = requests.post(url, files=files, headers=headers)

# Check the response status
if response.status_code == 200:
    response_data = response.json()
    image_id = response_data.get("id")
    print(f"Image uploaded successfully! Image ID: {image_id}")
else:
    print(f"Error uploading image. Status Code: {response.status_code}")
    print("Response:", response.text)  # Print the full response for debugging
