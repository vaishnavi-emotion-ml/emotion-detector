pip install pandas numpy seaborn scikit-learn neattext joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------
# 1. Sample Training Dataset
# --------------------------
more_data = {
    'text': [
        # happy
        "I feel amazing today!",
        "Life is beautiful.",
        "I'm full of joy.",
        # sad
        "I’m really down right now.",
        "Tears won’t stop falling.",
        "I'm feeling hopeless.",
        # angry
        "I can't believe they did that!",
        "I’m absolutely furious!",
        "They make me so mad.",
        # calm
        "I'm just relaxing right now.",
        "Everything feels so peaceful.",
        "Enjoying a moment of silence.",
        # fear
        "I’m terrified of the dark.",
        "That sound scared me.",
        "I’m feeling anxious again.",
        # love
        "You are my soulmate.",
        "I love being around you.",
        "My heart is full of love."
    ],
    'emotion': [
        "happy", "happy", "happy",
        "sad", "sad", "sad",
        "angry", "angry", "angry",
        "calm", "calm", "calm",
        "fear", "fear", "fear",
        "love", "love", "love"
    ]
}
df = pd.DataFrame(more_data)

extra_df = pd.DataFrame(more_data)
df = pd.concat([df, extra_df], ignore_index=True)


# ----------------------
# 2. Data Preprocessing
# ----------------------
df['clean_text'] = df['text'].apply(nfx.remove_userhandles)
df['clean_text'] = df['clean_text'].apply(nfx.remove_stopwords)

# --------------------------
# 3. Train/Test Split
# --------------------------
X = df['clean_text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# --------------------------
# 4. Model Pipeline
# --------------------------
emotion_model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

# --------------------------
# 5. Train Model
# --------------------------
emotion_model.fit(X_train, y_train)

# --------------------------
# 6. Evaluation
# --------------------------
y_pred = emotion_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --------------------------
# 7. Save Model
# --------------------------
joblib.dump(emotion_model, "emotion_model.pkl")

# --------------------------
# 8. User Input for Prediction
# --------------------------
user_text = input("Enter your text to analyze emotion: ")
user_text_clean = nfx.remove_stopwords(user_text)

# Predict emotion
predicted_emotion = emotion_model.predict([user_text_clean])[0]
print(f"\nPredicted Emotion: {predicted_emotion}")

# --------------------------
# 9. Show Prediction Probabilities
# --------------------------
probs = emotion_model.predict_proba([user_text_clean])[0]
emotion_labels = emotion_model.classes_

print("\nEmotion Probabilities:")
for emotion, prob in zip(emotion_labels, probs):
    print(f"{emotion}: {prob:.2f}")

# --------------------------
# 10. Plot Probabilities
# --------------------------
sns.set(style="whitegrid")
sns.barplot(x=emotion_labels, y=probs, palette="muted")
plt.title("Emotion Prediction Probabilities")
plt.ylabel("Confidence")
plt.xlabel("Emotion")
plt.ylim(0, 1)
plt.show()
