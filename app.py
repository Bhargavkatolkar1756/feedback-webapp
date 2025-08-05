from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load models only once
classifier = joblib.load("zero_shot_classifier.pkl")
summarizer = joblib.load("feedback_summarizer.pkl")

# Define feedback labels
labels = ["Bug", "Feature Request", "Complaint", "Compliment", "General Feedback"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        
        # Ensure input is not empty
        if not feedback_text.strip():
            return render_template('index.html', error="Please enter feedback text.")
        
        # Classification
        classification_result = classifier(feedback_text, labels)
        predicted_category = classification_result['labels'][0]
        
        # Summarization (truncate to 1024 chars for model limits)
        summary_result = summarizer(feedback_text[:1024])[0]['summary_text']
        
        return render_template(
            'index.html',
            feedback=feedback_text,
            category=predicted_category,
            summary=summary_result,
            success=True
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
