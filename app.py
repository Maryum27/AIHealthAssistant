import os
import re
import gradio as gr
from groq import Groq
import speech_recognition as sr
from fpdf import FPDF

# =======================================
# API KEY
# =======================================
os.environ["GROQ_API_KEY"] = "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"
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# =======================================
# STRICT MEDICAL-ONLY SYSTEM PROMPT
# =======================================
SYSTEM_PROMPT = """
You are a strictly medical-only AI health assistant.

RULES:
1. You ONLY answer questions related to:
   - symptoms
   - diseases and causes
   - possible diagnosis (informational only)
   - safe OTC medications
   - home-care guidance
   - when to seek urgent or emergency care

2. If user asks ANYTHING non-medical:
   Reply exactly → “I can only help with medical or health-related questions.”

3. Do NOT give:
   - prescription medication
   - harmful instructions
   - non-medical advice

4. Your answers MUST:
   - Ask clarifying questions when needed
   - Suggest possible causes (NOT a diagnosis)
   - Give safe OTC remedies
   - Tell when to seek emergency care
   - Use simple, empathetic language

End EVERY answer with:
“I am not a medical professional. This is general information only.”
"""

# =======================================
# CALL AGENT
# =======================================
def call_agent(history, text):
    if not text or len(text.strip()) == 0:
        return history, history

    history.append({"role": "user", "content": text})

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=msgs,
        temperature=0.3,
        max_tokens=1500
    )

    assistant_msg = resp.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_msg})

    return history, history

# =======================================
# VOICE → TEXT
# =======================================
def transcribe_audio(audio_path, history):
    if audio_path is None:
        return history, history

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    except:
        text = "Sorry, I could not understand your voice."

    return call_agent(history, text)

# =======================================
# CALL & CLEAR TEXTBOX
# =======================================
def call_agent_and_clear(history, text):
    history, formatted = call_agent(history, text)
    return history, formatted, ""

# =======================================
# PDF EXTRACTION LOGIC
# =======================================
SYMPTOM_HINTS = [
    "pain","ache","fever","cough","sore","nausea","vomit","diarrhea",
    "headache","dizzy","rash","itch","swelling","fatigue",
]

OTC_KEYWORDS = [
    "paracetamol","acetaminophen","ibuprofen","loratadine",
    "cetirizine","antacid","ORS","oral rehydration","saline spray"
]

CAUSE_KEYWORDS = ["possible", "likely", "may be", "could be"]

EMERGENCY_KEYWORDS = ["emergency", "urgent", "seek care", "call doctor"]

def extract_pdf_items_precise(history, disease):
    disease = disease.lower().strip()
    symptoms, meds, causes, emergency = [], [], [], []

    for msg in history:
        text = msg["content"]
        text_lower = text.lower()

        if msg["role"] == "user":
            if disease in text_lower or any(s in text_lower for s in SYMPTOM_HINTS):
                for s in SYMPTOM_HINTS:
                    if s in text_lower and text not in symptoms:
                        symptoms.append(text)

        if msg["role"] == "assistant":
            sentences = re.split(r'[.!?]\s*', text)
            for sent in sentences:
                sent_lower = sent.lower()
                if any(m in sent_lower for m in OTC_KEYWORDS):
                    if sent.strip() not in meds:
                        meds.append(sent.strip())
                if any(c in sent_lower for c in CAUSE_KEYWORDS):
                    if sent.strip() not in causes:
                        causes.append(sent.strip())
                if any(e in sent_lower for e in EMERGENCY_KEYWORDS):
                    if sent.strip() not in emergency:
                        emergency.append(sent.strip())

    return symptoms[:20], causes[:10], meds[:10], emergency[:10]

# =======================================
# PDF GENERATOR
# =======================================
def generate_pdf_report(history, disease):
    disease = disease.strip()
    if not disease:
        raise gr.Error("Please write the disease name.")

    symptoms, causes, meds, emergency = extract_pdf_items_precise(history, disease)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Patient Medical Summary", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Disease / Concern: {disease.title()}", ln=True)
    pdf.ln(4)

    # Symptoms
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Symptoms Reported:", ln=True)
    pdf.set_font("Arial", "", 12)
    if symptoms:
        for s in symptoms:
            pdf.multi_cell(0, 8, f"- {s}")
    else:
        pdf.multi_cell(0, 8, "- No symptoms detected.")
    pdf.ln(4)

    # Causes
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Possible Causes:", ln=True)
    pdf.set_font("Arial", "", 12)
    if causes:
        for c in causes:
            pdf.multi_cell(0, 8, f"- {c}")
    else:
        pdf.multi_cell(0, 8, "- No cause-related replies found.")
    pdf.ln(4)

    # Medications
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Possible OTC Medications:", ln=True)
    pdf.set_font("Arial", "", 12)
    if meds:
        for m in meds:
            pdf.multi_cell(0, 8, f"- {m}")
    else:
        pdf.multi_cell(0, 8, "- No OTC medications found.")
    pdf.ln(4)

    # Emergency
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Emergency Guidance:", ln=True)
    pdf.set_font("Arial", "", 12)
    if emergency:
        for e in emergency:
            pdf.multi_cell(0, 8, f"- {e}")
    else:
        pdf.multi_cell(0, 8, "- No emergency advice detected.")
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 7, "Disclaimer: I am not a medical professional. This is general information only.")

    filename = f"Report_{disease.replace(' ', '_')}.pdf"
    pdf.output(filename)

    return filename

# =======================================
# CUSTOM CSS
# =======================================
css = """
#buttons-row { display:flex; gap:12px; justify-content:center; }
button { border-radius:10px !important; height:48px !important; font-size:16px !important; }
#symptom-box { border-radius:12px !important; padding:12px !important; }
"""

# =======================================
# GRADIO UI
# =======================================
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("<h2 style='text-align:center;'>🩺 AI Health Assistant</h2>")

    chatbot = gr.Chatbot(height=450, type="messages")
    state = gr.State([])

    text_input = gr.Textbox(
        placeholder="Describe symptoms...",
        label="Your Message",
        elem_id="symptom-box"
    )

    with gr.Row(elem_id="buttons-row"):
        mic_btn = gr.Audio(label="🎤 Mic", type="filepath")
        send_btn = gr.Button("Send ➤", variant="primary")
        pdf_btn = gr.Button("📄 PDF Report", variant="secondary")

    disease_input = gr.Textbox(label="Enter Disease for PDF", placeholder="e.g., migraine")
    pdf_output = gr.File(label="Download PDF")

    mic_btn.change(transcribe_audio, inputs=[mic_btn, state], outputs=[state, chatbot])
    send_btn.click(call_agent_and_clear, inputs=[state, text_input], outputs=[state, chatbot, text_input])
    text_input.submit(call_agent_and_clear, inputs=[state, text_input], outputs=[state, chatbot, text_input])
    pdf_btn.click(generate_pdf_report, inputs=[state, disease_input], outputs=[pdf_output])

demo.launch()
