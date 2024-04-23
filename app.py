from flask import Flask, render_template, request
import random
import json
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('popular')
lemmatizer = WordNetLemmatizer()

model = load_model("models/chat_model.h5")
intents = json.loads(open('json/quires.json').read())
words = pickle.load(open('models/texts.pkl', 'rb'))
classes = pickle.load(open('models/labels.pkl', 'rb'))

app = Flask(__name__)
app.static_folder = 'static'
file1 = open("data.txt", "a")


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def compare_courses(course_data):
    best_course = None
    highest_rating = 0

    print(course_data)
    for course in course_data.get('courses'):
        print(course)
        rating = course['rating']
        if rating > highest_rating:
            best_course = course['name']
            highest_rating = rating

    return best_course, highest_rating


def load_courses_from_json(json_file):
    with open(json_file) as f:
        courses_data = json.load(f)
    return courses_data


def main():
    # Load course data from JSON file
    course_data = load_courses_from_json('json/courses.json')

    # Compare courses and find the best one
    best_course, highest_rating = compare_courses(course_data)

    print("The best course is:", best_course)
    print("Rating:", highest_rating)


if __name__ == "__main__":
    main()


def getResponse(ints, intents_json):
    print('GetResponse called with arguments: ', ints, ' ', intents_json)
    tag = ints[0]['intent']
    print('Tag: ', tag)
    list_of_intents = intents_json['intents']
    print('List of intents: ', list_of_intents)
    result = "Sorry, I didn't understand that."
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route("/")
def home():
    print('In home page')
    return render_template("index.html")


@app.route("/index")
def index():
    print('In Index page')
    return render_template("index.html")


@app.route("/get")
def chatbot_response():
    userText = request.args.get('msg')
    print(userText)
    # Get the predicted intent from the user input
    predicted_intents = predict_class(userText, model)

    # Get a response based on the predicted intent
    response = getResponse(predicted_intents, intents)

    return response


if __name__ == "__main__":
    app.run(debug=True, port=3000)
