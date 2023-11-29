import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from io import BytesIO
import speech_recognition as sr
from playsound import playsound
from openai import OpenAI
import singlestoredb as s2

# Set up OpenAI API key
# This line retrieves the API key from an environment variable for security.
OpenAI.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

# Create the financial assistant
# This section creates a file from a local JSON and then uses it to create an assistant in OpenAI.
file = client.files.create(
    file=open("portfolio.json", "rb"),
    purpose='assistants'
)

assistant = client.beta.assistants.create(
    name="Financial Assistant",
    instructions="You are a financial assistant. Use the data about the user's stock portfolio to give them advice about how their finance is doing. Each stock/index fund/crypto in the file has this information: [details]",
    model="gpt-4-1106-preview",
    tools=[{"type": "retrieval"}],
    file_ids=[file.id]
)

# Function to play audio
# This function uses OpenAI's Text-to-Speech model to convert text to speech and play it.
def play_audio(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.stream_to_file("audio/output.mp3")
    playsound("audio/output.mp3")

# Function to get transcript from audio
# This function converts spoken words (audio file) into text using OpenAI's Whisper model.
def get_prompt():
    with open("audio/input.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript['text']

# Function to record and save audio
# This function records audio from the microphone and saves it as an MP3 file.
def get_input_file(threshold=0.03, silence_duration=3):
    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        print("Listening for speech...")
        recognizer.adjust_for_ambient_noise(mic)
        started = False
        start_time = None
        audio_frames = []
        recording = True

        def callback(indata, frames, time, status):
            nonlocal started, start_time, audio_frames, recording
            if np.any(indata > threshold):
                if not started:
                    print("Starting recording...")
                    started = True
                    start_time = time.inputBufferAdcTime
                audio_frames.append(indata.copy())
            elif started:
                if time.inputBufferAdcTime - start_time > silence_duration:
                    recording = False
                    raise sd.CallbackAbort

        with sd.InputStream(callback=callback, channels=1):
            while recording:
                pass

        if audio_frames:
            audio_data = np.concatenate(audio_frames, axis=0)
            with BytesIO() as f:
                sf.write(f, audio_data, samplerate=70000, format='WAV')
                f.seek(0)
                with sr.AudioFile(f) as source:
                    audio = recognizer.record(source)
                    with open("audio/input.mp3", "wb") as mp3_file:
                        mp3_file.write(audio.get_wav_data(convert_rate=16000, convert_width=2))

# Main function to run the assistant
def main():
    # Connect to SingleStore database
    # This line establishes a connection to the SingleStore database.
    conn = s2.connect('admin:Testing123@svc-ca8fa339-0d39-4942-ad73-4463f4110a1c-dml.aws-virginia-5.svc.singlestore.com:3306/testing')
    conn.autocommit(True)

    try:
        while True:
            get_input_file()
            user_prompt = get_prompt()
            print("User said:", user_prompt)

            # Send the user's query to the financial assistant and get the response
            # This section sends the transcribed user query to the OpenAI assistant and retrieves the response.
            thread = client.beta.threads.create()
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_prompt,
                file_ids=[file.id]
            )
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            # Assuming the last message in the thread is the assistant's response
            assistant_response = messages[-1]['content']
            print("Assistant's response:", assistant_response)
            play_audio(assistant_response)

            # SQL statement for a regular INSERT
            # This section inserts the user's query into the SingleStore database.
            insert_stmt = 'INSERT INTO OpenAISingleStore (TextValue) VALUES (%s)'

            with conn.cursor() as cur:
                # Insert the data without specifying TextKey; it will auto-increment
                cur.execute(insert_stmt, (user_prompt,))

                # Retrieve the last inserted ID
                cur.execute('SELECT LAST_INSERT_ID()')
                last_id_result = cur.fetchone()
                if last_id_result:
                    last_id = last_id_result[0]
                    print("Last inserted ID:", last_id)

    finally:
        # Ensure the connection is closed after the loop
        conn.close()

if __name__ == "__main__":
    main()
