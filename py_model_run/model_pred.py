from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_model(checkpoint_path = "training_1/cp.ckpt", tokenizer=tokenizer):
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_reloaded = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_reloaded.load_weights(checkpoint_path)
    return model_reloaded

def get_predictions(text_list,model_reloaded,tokenizer=tokenizer):
    tf_batch = tokenizer(text_list, max_length=500, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model_reloaded(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    labels = ['terrible','bad','neutral','happy','very happy']
    sentiment_map = {0:'negative', 1:'negative', 2:'neutral', 3:'positive' ,4:'positive' }
    predictions = {} #format: text:[sentiment, categorization]
    for i in range(len(text_list)):
        predictions[text_list[i]] = [sentiment_map[label[i]],labels[label[i]]]

    return predictions


model = load_model()

app = Flask(__name__)
api = Api(app)

emails = {}

class Messages(Resource):
    def get(self, email_id):
        return {email_id: emails[email_id]}

    def put(self, email_id):
        emails[email_id]= request.form['data']
        l = []
        l.append(emails[email_id])
        result = get_predictions(l, model)
        return {email_id: result}

api.add_resource(Messages, '/<string:email_id>')

if __name__ == '__main__':
    app.run(debug=True)
=======
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import tensorflow as tf
import pandas as pd
from json import dumps


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_model(checkpoint_path = "training_1/cp.ckpt", tokenizer=tokenizer):
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_reloaded = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_reloaded.load_weights(checkpoint_path)
    return model_reloaded


def get_predictions(text_list,model_reloaded,tokenizer=tokenizer):
    tf_batch = tokenizer(text_list, max_length=128, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model_reloaded(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    labels = ['terrible','bad','neutral','happy','very happy']
    sentiment_map = {0:'negative', 1:'negative', 2:'neutral', 3:'positive' ,4:'positive' }
    predictions = {} #format: text:[sentiment, categorization]
    for i in range(len(text_list)):
        predictions[text_list[i]] = [sentiment_map[label[i]],labels[label[i]]]
    return predictions

model = load_model()
# # Send text in form of list
# pred_sentences = input('enter text to predict seperated by comma: ')
# pred_sentences = pred_sentences.split(',')
# pred_sentences = [i.strip() for i in pred_sentences]
# result = get_predictions(pred_sentences)
# print(result)

app = Flask(__name__)
api = Api(app)


class Messages(Resource):

    def post(self):
        print(request.json)
        Message = request.json['Message']
        result = get_predictions(Message)
        return {'label': result}


api.add_resource(Messages, '/messages')

if __name__ == '__main__':
    app.run()
