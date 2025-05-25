# Sign-Language-to-Text

Project converts American sign language to text in realtime. It uses CNN to train the required models for prediction. The dataset is custom made.

dataset : 
- train : https://drive.google.com/drive/u/1/folders/1-XTAjPPRPFeRqu3848z8dMXaolILWizn
- test : https://drive.google.com/drive/u/1/folders/18e1F1n1SWPF8lUF8pCKdUzSzKAbmSbVN

Demo : https://www.youtube.com/watch?v=aU5-8XJrxwY&t=2s

# 🐍 Setting Up Python Project on Raspberry Pi via RealVNC

Follow the steps below to set up and run your Python project on your Raspberry Pi using RealVNC:

---

## ✅ Steps

1. **Connect to Raspberry Pi**
   - Use **RealVNC Viewer** to connect to your Raspberry Pi over the same Wi-Fi network.

2. **Open Terminal**
   - Once connected, open the terminal **inside the VNC window**, not from VS Code.

3. **Navigate to Project Directory**
   ```bash
   cd /path/to/your/project
   ```

4. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   ```

5. **Activate the Virtual Environment**
   ```bash
   source venv/bin/activate
   ```

6. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Verify Installations**
   ```bash
   python test_imports.py
   ```

8. **Run the Main Application**
   - After verifying, run your main app as needed:
     ```bash
     python main.py
     ```

---

📌 **Note:** Replace `/path/to/your/project` with the actual path to your project folder.

## ⚠️ Important Notes

- Make sure you have **Python 3.7 or higher** installed on your Raspberry Pi.
- If you encounter any **memory issues**, you might need to **increase the swap space**.
- Some packages might take **longer to install on Raspberry Pi** due to compilation requirements.
- If you face any **specific errors**, feel free to share them — I'm happy to help you resolve them.



