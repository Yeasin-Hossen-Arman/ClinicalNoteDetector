from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import io
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load dataset and train model
DATA_PATH = 'clinical_non_clinical_dataset.csv'

# Load data
df = pd.read_csv(DATA_PATH)
texts = df['text'].tolist()
labels = df['label'].tolist()

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = None
    clinical_percent = None
    non_clinical_percent = None
    show_download = False

    if request.method == 'POST':
        input_text = request.form['input_text']
        X_input = vectorizer.transform([input_text])
        prob = model.predict_proba(X_input)[0]

        clinical_percent = round(prob[1] * 100, 2)
        non_clinical_percent = round(prob[0] * 100, 2)

        if clinical_percent > 50:
            show_download = True

    return render_template('index.html', 
                           prediction=prediction,
                           input_text=input_text,
                           clinical_percent=clinical_percent,
                           non_clinical_percent=non_clinical_percent,
                           show_download=show_download)

# Route to download PDF
@app.route('/download', methods=['POST'])
def download_pdf():
    input_text = request.form['input_text']
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 12)

    # Write text into PDF, split by lines
    lines = input_text.split('\n')
    y = 800
    for line in lines:
        p.drawString(50, y, line)
        y -= 20
        if y < 50:
            p.showPage()
            y = 800
    
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="clinical_note.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # 5000 is a fallback for local runs
    app.run(host='0.0.0.0', port=port)
