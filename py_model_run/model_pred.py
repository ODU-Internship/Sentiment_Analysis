from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd

def load_model(checkpoint_path = "training_1/cp.ckpt"):
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_reloaded = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_reloaded.load_weights(checkpoint_path)
    return model_reloaded

def get_predictions(text_list):
    tf_batch = tokenizer(pred_sentences, max_length=500, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model_reloaded(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    labels = ['terrible','bad','neutral','happy','very happy']
    sentiment_map = {0:'negative', 1:'negative', 2:'neutral', 3:'positive' ,4:'positive' }
    predictions = {} #format: text:[sentiment, categorization]
    for i in range(len(pred_sentences)):
        predictions[pred_sentences[i]] = [sentiment_map[label[i]],labels[label[i]]]

    return predictions


model = load_model()
pred_sentences = input('enter text to predict seperated by comma: ') #Send text in form of list
pred_sentences = pred_sentences.split(',')
pred_sentences = [ i.strip() for i in pred_sentences]
result = get_predictions(pred_sentences)
print(result)