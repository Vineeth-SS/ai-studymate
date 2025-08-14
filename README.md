# 📚 StudyMate — AI-Powered PDF Q&A

## Description
StudyMate is a **Streamlit-based study assistant** that lets you upload one or more PDF documents, ask natural language questions about them, and get accurate, context-based answers powered by a **local Ollama model** (e.g., `granite3.3:8b`). It also includes a **notes management system** that saves your Q/A history with timestamps for each PDF, so you can review and study later — without having to reopen files or lose context.

---

## ✨ Key Features
- 📂 **Multiple PDF Uploads** — Manage and switch between multiple documents in one session.  
- 💬 **Interactive Q&A Chat** — Ask questions about the selected PDF; answers come directly from the model using PDF context.  
- 🗂 **Per-PDF Conversations** — Each uploaded file has its own separate Q&A history.  
- 📝 **Notes Saving with Timestamps** — Automatically saves all Q/A in a timestamped `.txt` file for that PDF.  
- 👀 **In-App Notes Viewer** — View your saved notes directly in the UI without opening files manually.  
- 🧠 **Vector Search** — Splits PDFs into chunks, embeds them, and retrieves the most relevant parts for each query.  
- 🖌 **Clean UI** — Styled chat bubbles, intuitive layout, and easy-to-use controls.

---

## 🚀 How to Use

### 1️⃣ Install dependencies
```bash
pip install streamlit PyPDF2 sentence-transformers ollama faiss-cpu beautifulsoup4
```
Make sure **Ollama** is installed and running locally with your desired model pulled:  
```bash
ollama pull granite3.3:8b
```

### 2️⃣ Run the app
```bash
streamlit run app.py
```

### 3️⃣ Upload PDFs
- Drag-and-drop or browse to select **one or more PDF files**.

### 4️⃣ Select a PDF & Chat
- Pick a document from the dropdown.  
- Type a question and click **Get Answer**.  
- The AI will search inside the PDF for relevant text and respond.

### 5️⃣ View & Save Notes
- All Q/A pairs are automatically saved with timestamps in the `notes/` folder.  
- Notes for the selected PDF are displayed inside the app.

### 6️⃣ Switch Documents
- Change the dropdown to switch between PDFs and their separate notes.

---

## 🏆 Ideal Use Cases
This setup is **ideal for students, researchers, and professionals** who want to quickly extract information from large PDFs and keep an organized Q/A study log.

---

## 📌 Example Workflow
1. Upload 3 PDFs (`research_paper.pdf`, `course_notes.pdf`, `manual.pdf`).
2. Ask targeted questions for each PDF separately.
3. Review saved notes in-app with timestamps for reference.
4. Export or use notes for revision, reports, or research documentation.

---

## 📄 License
This project is open-source under the MIT License.
