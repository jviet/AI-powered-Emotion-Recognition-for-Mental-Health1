# AI-powered-Emotion-Recognition-for-Mental-Health1
開発AI駆動の感情認識ツールは、ユーザーの感情状態を分析し、メンタルヘルスサポートを提供します。
from transformers import pipeline
import textwrap

# Setup the emotion classification pipeline
emotion_classifier = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

def analyze_emotion(text: str):
    """
    Analyze the emotion of the given text and return the dominant emotion.

    Parameters:
    - text (str): User input text to analyze.

    Returns:
    - str: The dominant emotion detected in the text.
    """
    results = emotion_classifier(text)
    emotion_scores = {result['label']: result['score'] for result in results[0]}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    return dominant_emotion, emotion_scores

def provide_support(dominant_emotion: str):
    """
    Provides mental health support advice based on the dominant emotion detected.

    Parameters:
    - dominant_emotion (str): The detected dominant emotion.

    Returns:
    - str: Suggested support advice.
    """
    advice = {
        'sadness': "It seems you're feeling sad. It's okay to feel this way. Consider talking to a friend or professional.",
        'joy': "It's great to see you're feeling happy! Remember what made you feel this way and cherish it.",
        'anger': "Anger can be overwhelming. Try to take a few deep breaths and think of a peaceful place.",
        'fear': "Feeling scared is a natural response. Reflecting on what's causing this fear may help in understanding it better.",
        'surprise': "Surprises can be shocking. Give yourself a moment to process this feeling.",
        'disgust': "Feeling disgusted can be uncomfortable. Consider why this feeling arose and if there's a way to address it.",
        'neutral': "Feeling neutral is perfectly fine. Sometimes, not feeling much can be a moment of peace."
    }
    return advice.get(dominant_emotion, "It's important to acknowledge your feelings. Consider speaking to someone who can help.")

def main():
    """
    Main function to run the emotion recognition and provide mental health support.
    """
    user_input = input("How are you feeling today? Please share your thoughts: ")
    dominant_emotion, scores = analyze_emotion(user_input)
    print(f"\nDetected Emotion: {dominant_emotion.capitalize()}")
    print("Emotion Scores:", scores)
    advice = provide_support(dominant_emotion)
    print("\nMental Health Support Advice:")
    print(textwrap.fill(advice, width=80))

if __name__ == "__main__":
    main()
