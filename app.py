import streamlit as st
import sqlite3
import pandas as pd
import os
import tempfile
from datetime import datetime
import zipfile
import re

# transformer model
from transformers import BartTokenizerFast, BartForConditionalGeneration

# optional libs
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import moviepy.editor as mp
except Exception:
    mp = None


# ------------------------------
# Config & Model Loading
# ------------------------------
MODEL_DIR = "./bart_finetuned"  # path to your saved fine-tuned model


@st.cache_resource
def load_model():
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
    model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
    return tokenizer, model


try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Failed to load model from {MODEL_DIR}: {e}")
    st.stop()


# ------------------------------
# Database helpers
# ------------------------------
DB_PATH = "summaries.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        content TEXT,
        summary TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("PRAGMA table_info(summaries)")
    cols = [row[1] for row in cursor.fetchall()]
    if "filename" not in cols:
        cursor.execute("ALTER TABLE summaries ADD COLUMN filename TEXT;")
    if "content" not in cols:
        cursor.execute("ALTER TABLE summaries ADD COLUMN content TEXT;")
    if "summary" not in cols:
        cursor.execute("ALTER TABLE summaries ADD COLUMN summary TEXT;")
    if "created_at" not in cols:
        cursor.execute("ALTER TABLE summaries ADD COLUMN created_at TEXT;")

    conn.commit()
    conn.close()


def save_summary(filename, content, summary):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO summaries (filename, content, summary, created_at) VALUES (?, ?, ?, ?)",
        (filename, content, summary, created_at)
    )
    conn.commit()
    conn.close()


def fetch_all_summaries():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM summaries ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_summary(summary_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
    conn.commit()
    conn.close()


init_db()


# ------------------------------
# Cleaning helper
# ------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)  # strip weird chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------
# Extraction helpers
# ------------------------------
def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def extract_text_from_docx(path):
    if docx2txt is None:
        raise RuntimeError("docx2txt not installed")
    return clean_text(docx2txt.process(path))


def extract_text_from_pdf(path):
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not installed")
    reader = PyPDF2.PdfReader(path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            pages.append(txt)
    return clean_text("\n".join(pages))


def extract_text_from_image(path):
    if pytesseract is None or Image is None:
        raise RuntimeError("pytesseract or Pillow not installed")
    img = Image.open(path)
    return clean_text(pytesseract.image_to_string(img))


def transcribe_audio_file(path):
    if sr is None:
        raise RuntimeError("speech_recognition not installed")
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_data = r.record(source)
    return clean_text(r.recognize_google(audio_data))


def extract_text_from_video(path):
    if mp is None:
        raise RuntimeError("moviepy not installed")
    tmp_wav = path + ".wav"
    clip = mp.VideoFileClip(path)
    if clip.audio is None:
        raise RuntimeError("Video has no audio track")
    clip.audio.write_audiofile(tmp_wav, logger=None)
    text = transcribe_audio_file(tmp_wav)
    try:
        os.remove(tmp_wav)
    except Exception:
        pass
    return text


def extract_text_from_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    max_rows = 50
    if len(df) > max_rows:
        df = df.head(max_rows)

    lines = []
    for _, row in df.iterrows():
        row_str = "; ".join(f"{col}: {str(val)}" for col, val in row.items())
        lines.append(f"- {row_str}")

    return clean_text("\n".join(lines))


def extract_text_from_zip(path, tmp_dir):
    results = []
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(tmp_dir)
        for fname in z.namelist():
            fpath = os.path.join(tmp_dir, fname)
            if os.path.isdir(fpath):
                continue
            try:
                with open(fpath, "rb") as f:
                    class DummyUpload:
                        def __init__(self, name, data):
                            self.name = name
                            self._data = data

                        def read(self):
                            return self._data

                    fake_upload = DummyUpload(fname, f.read())
                text = extract_text_from_uploaded_file(fake_upload, tmp_dir)
                if isinstance(text, list):
                    results.extend(text)
                else:
                    results.append((fname, text))
            except Exception as e:
                results.append((fname, f"[ERROR extracting {fname}: {e}]"))
    return results


def extract_text_from_uploaded_file(uploaded_file, tmp_dir):
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()
    tmp_path = os.path.join(tmp_dir, name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    if ext == "zip":
        return extract_text_from_zip(tmp_path, tmp_dir)
    if ext == "txt":
        return extract_text_from_txt(tmp_path)
    if ext == "docx":
        return extract_text_from_docx(tmp_path)
    if ext == "pdf":
        return extract_text_from_pdf(tmp_path)
    if ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
        return extract_text_from_image(tmp_path)
    if ext in ("wav", "flac", "aiff", "aif", "mp3", "m4a", "aac", "ogg"):
        return transcribe_audio_file(tmp_path)
    if ext in ("mp4", "mov", "avi", "mkv"):
        return extract_text_from_video(tmp_path)
    if ext == "csv":
        return extract_text_from_csv(tmp_path)

    raise RuntimeError(f"Unsupported file extension: {ext}")


# ------------------------------
# Summarization helper
# ------------------------------
def summarize_text(text, max_input_len=512, max_out=128, min_out=20):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "[No text extracted]"
    truncated = text if len(text) <= 4000 else text[:4000]
    inputs = tokenizer([truncated], max_length=max_input_len,
                       truncation=True, return_tensors="pt")
    ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_out,
        min_length=min_out,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Universal Summarizer", layout="wide")
st.title("🌐 Universal Summarizer — Text / Audio / Video / Image / ZIP / CSV")

left, right = st.columns([2, 3])

with left:
    st.header("Upload / Paste Content")
    input_mode = st.radio("Input mode", ["Paste Text", "Upload Files"], index=1)

    files_to_process = []

    if input_mode == "Paste Text":
        pasted = st.text_area("Paste text here:", height=300)
        filename_for_paste = st.text_input("Filename label (optional):", value="pasted_text.txt")
        if pasted and st.button("Summarize Pasted Text"):
            summary = summarize_text(pasted)
            st.subheader("Summary")
            st.success(summary)
            save_summary(filename_for_paste or "pasted_text", pasted, summary)

    else:
        uploaded_files = st.file_uploader(
            "Upload files (txt, docx, pdf, png/jpg, wav/mp3, mp4, csv, zip, ...)",
            type=None,
            accept_multiple_files=True
        )
        if uploaded_files:
            tmp_dir = tempfile.mkdtemp(prefix="uploads_")
            st.info(f"Saved temporary files to {tmp_dir}")
            to_show = []
            for uf in uploaded_files:
                try:
                    with st.spinner(f"Extracting from {uf.name} ..."):
                        result = extract_text_from_uploaded_file(uf, tmp_dir)
                        if isinstance(result, list):  # zip → multiple files
                            for fname, text in result:
                                if text and len(text.strip()) > 0:
                                    files_to_process.append((fname, text))
                                    to_show.append((fname, len(text)))
                                else:
                                    to_show.append((fname, 0))
                        else:
                            if result and len(result.strip()) > 0:
                                files_to_process.append((uf.name, result))
                                to_show.append((uf.name, len(result)))
                            else:
                                to_show.append((uf.name, 0))
                except Exception as e:
                    to_show.append((uf.name, f"ERROR: {e}"))

            st.markdown("**Extraction results:**")
            for name, info in to_show:
                st.write(f"- **{name}** → {info if isinstance(info, str) else str(info)+' chars'}")

            if files_to_process and st.button("Summarize All Uploaded"):
                for fname, text in files_to_process:
                    with st.spinner(f"Summarizing {fname} ..."):
                        summary = summarize_text(text)
                    st.markdown(f"### 📌 Summary — **{fname}**")
                    st.write(summary)
                    save_summary(fname, text, summary)
                st.success("All uploaded files summarized and saved.")

with right:
    st.header("Saved Summaries (Database)")
    rows = fetch_all_summaries()
    if rows:
        df = pd.DataFrame(rows, columns=["ID", "Filename", "Content", "Summary", "Created At"])
        display_cols = ["ID", "Filename", "Created At", "Summary"]

        st.dataframe(df[display_cols], height=300)

        selected = st.selectbox("Select ID to view details", df["ID"].tolist())
        if selected:
            sel_row = df[df["ID"] == int(selected)].iloc[0]
            st.markdown(f"**Filename:** {sel_row['Filename']}")
            st.markdown(f"**Created At:** {sel_row['Created At']}")
            st.markdown("**Extracted Content (preview)**")
            st.text_area("Content", sel_row["Content"][:4000], height=220)
            st.markdown("**Summary**")
            st.text_area("Summary", sel_row["Summary"], height=150)

            if st.button("❌ Delete Selected"):
                delete_summary(int(selected))
                st.success(f"Deleted ID {selected}")
                st.rerun()

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download all as CSV", data=csv, file_name="summaries_all.csv", mime="text/csv")

        try:
            excel_path = os.path.join(tempfile.gettempdir(), "summaries_all.xlsx")
            df.to_excel(excel_path, index=False, engine="openpyxl")
            with open(excel_path, "rb") as f:
                st.download_button("📊 Download all as Excel (.xlsx)", data=f, file_name="summaries_all.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.warning(f"Excel export not available: {e}")

    else:
        st.info("No summaries saved yet. Upload files or paste text and summarize.")

st.markdown("---")
st.caption("Built with Hugging Face Transformers & Streamlit — Summaries stored in SQLite")
