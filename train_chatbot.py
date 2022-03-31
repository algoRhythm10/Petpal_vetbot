import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
import nltk
import random
import json
import pickle
from nltk.stem import WordNetLemmatizer

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import gradient_descent_v2


lemmatizer = WordNetLemmatizer()

words=[]
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']
data_file = open('C:\\Users\\acer\\Desktop\\VetBot\\intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        #add documents in the corpus
        documents.append((word_list, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word_list.lower()) for word_list in words if word_list not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []

output_empty = [0] * len(classes)

# training set bag of words for each sentence
for doc in documents:

    bag = []

    pattern_words = doc[0]
    # lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # create our bag of words array with 1, if word match found in current pattern
    for word_list in words:
        bag.append(1) if word_list in pattern_words else bag.append(0)
    
    # output is '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# create train and test lists (X - patterns, Y - intents)

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data is created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to no. of intents 

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = gradient_descent_v2.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


hist = model.fit(np.array(train_x), np.array(train_y), epochs=700, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("VetBot_Model is created")
