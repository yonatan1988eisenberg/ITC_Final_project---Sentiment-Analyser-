import pickle

import numpy as np
from flask import Flask
from flask import request
import os
import sklearn

app = Flask(__name__)



@app.route('/')
def input_ins():
    return None

def return_most_imp_words_in_texts(texts_tuple, top_200):
    '''
    Return the words in the reviews which are in the 200 most important words
    '''
    
    imp_words_texts = []
    for text in texts_tuple:
        imp_words = []
        for word in top_200:
            if word in text.split():
                imp_words.append(word)
        imp_words_texts.append(imp_words)

    return imp_words_texts

@app.route('/sentiment_review')
def pred_output():

    model = pickle.load(open('save.p', 'rb'))  # load model

    texts_tuple = eval(request.args.get("tuple_of_texts"))   # get texts inputs as a tuple
    
    with open('top_200_imp_words.txt') as f:
        top_200 = f.read()  # load the 200 most important words for the model

    top_200 = top_200.split('\n')


    imp_words_texts = return_most_imp_words_in_texts(texts_tuple, top_200)  # Get the most important words for each review

    output = model.predict(texts_tuple)  # model classification output
    most_imp_words_and_preds = zip(output, imp_words_texts)  # put most important words and model output together
    most_imp_words_and_preds_list = list(most_imp_words_and_preds)
    return f'{most_imp_words_and_preds_list}'



if __name__ == '__main__':


    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))

    else:
        app.run()
