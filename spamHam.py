# spam ham
# import tfid vectorizer,regular exp,bayes algo

from sklearn.feature_extraction.text import TfidfVectorizer
import re
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


#read file and load data

def loadData(fileName):
    fh = open(fileName)
    data = []
    label=[]
    pattern = re.compile(r'(.+)(\t)(.+)')
    # print(pattern)
    for msg in fh:
        x=re.search(pattern,msg)
        if x:
            label.append(x.group(1))
            data.append(x.group(3))
    fh.close()
    return data,label


data,label = loadData('./smsData')

#divide test and train dataset
# print("Size of data",len(data))
# print(data[0])
train_m,test_m,train_l,test_l = train_test_split(data,label,test_size=0.2)
# print(len(data),len(test_m))
# print(len(train_l),len(test_l))

# use tfid vectorizer and then transform to create bag words
tVect = TfidfVectorizer(stop_words="english")
tVect.fit(train_m)
bow = tVect.transform(train_m)


#train model on bag words of training
algo = MultinomialNB()
algo.fit(bow,train_l)
print("Training Complete....")

#create bow of test dataset
for lbl,msg in zip(test_l,test_m):
    sample = tVect.transform([msg])
    result=algo.predict(sample)
    print(result[0],lbl,sep=":")

#get Output