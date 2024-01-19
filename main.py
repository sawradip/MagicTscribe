import gradio as gr
import yt_dlp
import os
import shutil
import torch
import zipfile
from pathlib import Path
from transformers import pipeline

# Define the directory for downloaded audio and transcriptions
download_dir = Path("downloads")
# Check if the directory exists
if download_dir.exists() and download_dir.is_dir():
    # Delete the directory
    shutil.rmtree(download_dir)
# Create the directory
download_dir.mkdir(parents=True, exist_ok=True)
    
    
transcript_dir = Path("transcripts")
# Check if the directory exists
if transcript_dir.exists() and transcript_dir.is_dir():
    # Delete the directory
    shutil.rmtree(transcript_dir)
# Create the directory
transcript_dir.mkdir(parents=True, exist_ok=True)
# download_dir.mkdir(exist_ok=True)
# transcript_dir.mkdir(exist_ok=True)

IS_OLD_GPU = True
asr_transcriber = pipeline(
                "automatic-speech-recognition",
                model="distil-whisper/large-v2", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
                torch_dtype=torch.float16,
                device="cuda:0", # or mps for Mac devices
                model_kwargs={"attn_implementation":"flash_attention_2",
                            #   "use_flash_attention_2":is_flash_attn_2_available()},# True} #
                } if not IS_OLD_GPU else None,
            )

                
def download_audio(url):
    id = url.split("=")[-1]
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(download_dir / f'{id}'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_id = info.get('id', None)
        print(id, file_id)
        return id #file_id

def transcribe_audio(file_id):
    # Placeholder for the transcription function
    # You should implement this function based on your speech-to-text library
    # This function should return the transcription text of the given audio file
    file_pth = str(download_dir / f'{file_id}.mp3')
    tscription = asr_transcriber(inputs = file_pth,
                                return_timestamps=False,
                                batch_size = 16,
                                )["text"]
    return tscription

def process_urls(url_file):
    # Read the file and process each URL
    with open(url_file) as file_obj:
        urls = file_obj.read().splitlines()
    zip_filename = "transcriptions.zip"

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for url in urls:
            file_id = download_audio(url.strip())
            if file_id:
                transcription = transcribe_audio(file_id)
                transcript_file = transcript_dir / f"{file_id}.txt"
                with open(transcript_file, "w") as f:
                    f.write(transcription)
                zipf.write(transcript_file, arcname=f"{file_id}.txt")

    return zip_filename

iface = gr.Interface(
    fn=process_urls,
    inputs=gr.File(label="Upload Text File with YouTube URLs"),
    outputs=gr.File(label="Download Transcribed Texts"),
    allow_flagging="never"
)

iface.launch()
