import streamlit as st
import torch
import soundfile as sf
import librosa
from transformers import BartForConditionalGeneration, BartTokenizerFast, pipeline

# --------------------------
# Load Whisper ASR (HF pipeline)
# --------------------------
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# --------------------------
# Load Fine-Tuned BART
# --------------------------
MODEL_PATH = "C:\\Users\\saisr\\OneDrive\\文档\\project\\bart_finetuned"   # change if your model folder is different
tokenizer = BartTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

# --------------------------
# Helper: Transcribe + Summarize
# --------------------------
def summarize_audio(file):
    # 1. Load wav and force resample to 16kHz
    audio, sr = sf.read(file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 2. Whisper transcription
    transcript = asr({"array": audio, "sampling_rate": sr})["text"]

    # 3. Summarization with fine-tuned BART
    inputs = tokenizer(transcript, max_length=512, truncation=True, return_tensors="pt")
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return transcript, summary


# --------------------------
# Streamlit UI
# --------------------------
st.title("🎙️ Audio Summarizer (Whisper + BART)")

uploaded_audio = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav")
    
    with st.spinner("Transcribing and summarizing... ⏳"):
        transcript, summary = summarize_audio(uploaded_audio)

    st.subheader("Transcript")
    st.write(transcript)

    st.subheader("Summary")
    st.success(summary)
