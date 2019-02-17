import string

from nltk.corpus import stopwords
import pandas as pd
import random as rn
import numpy as np
from collections import Counter
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




"""
Cleaning the data removing the stopwords, apostrophe 's and low occuring words
including few high occuring words. Only one class is created for convinience of
coding. 
"""

class DataProcess:
    
    def cleandata(self, inputfile, val):
        b = str.maketrans('', '', string.punctuation)
        mostOccuring_words = ['film', 'movie', 'one', 'like', 'story']
        stopword = stopwords.words('english')
        with open(inputfile, 'r') as rawFile:
            File = rawFile.readlines()
            listofsentences = []
            sentences = []
            for sentence in File:
                #sentenceVer1 = [word for word in sentence.split() if word not in string.punctuation and len(word) > 3]
                sentenceVer1 = [word for word in sentence.lower().replace("'s", '').translate(b).split() if word not in string.punctuation and len(word)>2]
                sentenceVer2 = [word for word in sentenceVer1 if word not in stopword]
                sentenceVer3 = [word for word in sentenceVer2 if word not in mostOccuring_words]
                listofsentences.append([sentenceVer3, val])
            
            
            return listofsentences
                        
            #print(listofsentences)
            flat_list = [item for sublist in listofsentences for item in sublist] 
            
            wordcount = Counter(flat_list)            
            lowfrequencywords = [key for key, val in wordcount.items() if val == 1]
            #print(lowfrequencywords)
            rawFile.seek(0)
            for oneSentence in listofsentences:
                
                sentenceVer4 = ' '.join(word for word in oneSentence if word not in lowfrequencywords)
                sentences.append([sentenceVer4, val])
                
            return sentences

#creating array of document and combining them
                
    def arrayofdocument(self, inputfile):
        txtfile = []
        with open(inputfile, 'r') as file:
            for line in file:
                txtfile.append(line)
        return txtfile
        

   
    def combineArrayDocument(self, list1, list2):
        listofData = []
        for a in range(len(list1)):
            listofData.append(list1[a])
        for b in range(len(list2)):
            listofData.append(list2[b])
        rn.shuffle(listofData)
        return listofData

#calculate total words 
    
    def calculateTotalWords(self, fullDocumentList):
        count = 0
        for sentence in fullDocumentList:
            for words in sentence:
                    count+=1
        print(count)
        return count
 
#calculate Unique words
       
    def calculateUniqueWords(self, fullDocumentlist):
        listofwords = {}
        for sentence in fullDocumentlist:
            for words in sentence:
                if words in listofwords:
                    listofwords[words] += 1
                else:
                    listofwords[words] = 1
        return len(listofwords)

#calculate class probability
                    
    def calculateClassProbability(self, documentList, fullDocumentList):
        fullDCount = 0
        DCount = 0
        for sentence in fullDocumentList:
            fullDCount+=1
        for sentence in documentList:
            DCount+=1
        classProbability = DCount/fullDCount
        return classProbability

#creating probability dictionary
    
    def createProbabilityDictionary(self, documentList, d_totalWords, total_uniquewords):
        wordProbability_dict = {}
        sentenceList = []
        for sentences in documentList:
            for word in sentences:
                sentenceList.append(word)
           
        wordcount_dict = Counter(sentenceList)
        #print(wordcount_dict)
        
        for key, value in wordcount_dict.items():
            c_probability = (value + 1)/(d_totalWords + total_uniquewords)
            wordProbability_dict[key] = c_probability
            
        return wordProbability_dict

        
    def getProbability(self, word, dictionary, totalwords, uniquewords):
        value = dictionary.get(word, (1/(totalwords + uniquewords)))
        return value
        

    
    def eachSentenceProbability(self, classDict, documentLists, d_totalWords, total_uniquewords, classProb):
        
        SentenceProb_dict = OrderedDict()
        SentenceList = []
        
        for sentence in documentLists:
            #print(sentence)
            probability_pos = 1
            for words in sentence:
                if words in classDict:
                    prob_pos = classDict[words]
                    probability_pos *= prob_pos  
                else:
                    prob_pos = (1/(d_totalWords + total_uniquewords))
                    probability_pos *= prob_pos
                onesentence = " ".join(sentence)
                SentenceList.append(onesentence)
        
            SentenceProb_dict[onesentence] = classProb * probability_pos
        #print(SentenceProb_dict)
        return SentenceProb_dict
      
        
    def makePrediction(self, posSentDict, negSentDict):
        predictionList = []
        probabilityList = []
        count = 0
        for key in posSentDict:
            if posSentDict[key] > negSentDict[key]:
                predictionList.append(1)
                count+=1
                probabilityList.append(posSentDict[key])
            else:
                predictionList.append(0)
                count+=1
                probabilityList.append(posSentDict[key])
        print('prediction count: ', count)
        
        return (predictionList, probabilityList)
    
    #def shapeofList(targetsize, listsize)
    #def calculateAccuracy(self, )                    
        
    def converttodataframe(self, datalist):
        df_listofData = pd.DataFrame(datalist, columns = ['review', 'label'])
        return df_listofData
        
    def splitData(self, datalist, ratio):
        x = np.random.rand(len(datalist)) < ratio
        train = datalist[~x]
        train = train.reset_index(drop=True)
        test = datalist[x]
        test = test.reset_index(drop=True)
        return (train, test)
        

    def calculateConditionalProbabilityPos(self, dictionary, givenWord, uniqueWords):
        #positiveProbabilityDictionary = {}
        for givenword in dictionary:
            cProbability = (dictionary[givenword]+1)/(sum(dictionary.values())+uniqueWords)
            return cProbability
        
    
    def gettoptenwords(self, dictionary):
        #print(dictionary)
        word = []
        probability = []
        sorted_words = sorted(dictionary, key=dictionary.get, reverse=True)[:10]
        #print(sorted_words)
        for key in sorted_words:
            count = dictionary.get(key)
            word.append(key)
            probability.append(count)
        return (word, probability)
            
            
    
                
def main():
    DP = DataProcess()
    testDataPos = DP.cleandata("rt-polarity.pos.txt", 1)
    testDataNeg = DP.cleandata("rt-polarity.neg.txt", 0)
    testfullData = DP.combineArrayDocument(testDataPos, testDataNeg)
    df_testfullData = DP.converttodataframe(testfullData)
    print(df_testfullData.isnull().values.any())
    (df_trainNotReady, df_testReady) = DP.splitData(df_testfullData, 0.15)
    (df_trainReady, df_valReady) = DP.splitData(df_trainNotReady, 0.15)
    print(df_testfullData['review'].shape)
    print(df_testfullData['label'].shape)
    #trainFullData, trainPosData, trainNegData
    
    trainFullData = df_trainReady['review'].tolist()
    
    y_targetFulltrainData = df_trainReady['label'].tolist()
    print(len(y_targetFulltrainData))
    print(len(trainFullData))
    
    df_trainPosWithLabel = df_trainReady.loc[df_trainReady['label'] == 1]
    trainPosData = df_trainPosWithLabel['review'].tolist()
    df_trainNegWithLabel = df_trainReady.loc[df_trainReady['label'] == 0]
    trainNegData = df_trainNegWithLabel['review'].tolist()
    
    valData = df_valReady['review'].tolist()
    yTarget_valData = df_valReady['label'].tolist()
    testData = df_testReady['review'].tolist()
    yTarget_testData = df_testReady['label'].tolist()
    
        
    
    trainTotalpos = DP.calculateTotalWords(trainPosData)
    trainTotalneg = DP.calculateTotalWords(trainNegData)
    
    trainposClassProbability = DP.calculateClassProbability(trainPosData, trainFullData)
    trainnegClassProbability = DP.calculateClassProbability(trainNegData, trainFullData)
    
    trainuniqueWords = DP.calculateUniqueWords(trainFullData)
    trainpos_probability_Dict = DP.createProbabilityDictionary(trainPosData, trainTotalpos, trainuniqueWords)
    trainneg_probability_Dict = DP.createProbabilityDictionary(trainNegData, trainTotalneg, trainuniqueWords)
    
    trainpos_sentence_prob_dict = DP.eachSentenceProbability(trainpos_probability_Dict, valData, trainTotalpos, trainuniqueWords, trainposClassProbability)
    trainneg_sentence_prob_dict = DP.eachSentenceProbability(trainneg_probability_Dict, valData, trainTotalneg, trainuniqueWords, trainnegClassProbability)
    
    (predictions, probability) = DP.makePrediction(trainpos_sentence_prob_dict, trainneg_sentence_prob_dict)
    #print(predictions)
    print()
    print('Validation Data')
    print()    
    accuracy = accuracy_score(yTarget_valData, predictions)
    print('Accuracy: ', accuracy)
    F1Score = f1_score(yTarget_valData, predictions)
    print('F1Score :', F1Score)
    Con_Matrix = confusion_matrix(yTarget_valData, predictions)
    print(Con_Matrix)
   
    fpr, tpr, thresholds = roc_curve(yTarget_valData, predictions, pos_label=1)
    #print('False Positive Rate:', fpr)
    #print('True Positive Rate: ', tpr)
    print('thresholds:', thresholds)
    AUC = auc(fpr, tpr)
    plt.figure(1)
    plt.title('ROC for Validation Data')
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show
    print("Area Under the Curve: ", AUC)
    print()

    
    testpos_sentence_prob_dict = DP.eachSentenceProbability(trainpos_probability_Dict, testData, trainTotalpos, trainuniqueWords, trainposClassProbability)
    testneg_sentence_prob_dict = DP.eachSentenceProbability(trainneg_probability_Dict, testData, trainTotalneg, trainuniqueWords, trainnegClassProbability)
    
    (prediction, probabilities) = DP.makePrediction(testpos_sentence_prob_dict, testneg_sentence_prob_dict)
    #print(predictions)
    print('Test Data')
    print()    
    accuracy = accuracy_score(yTarget_testData, prediction)
    print('Accuracy: ', accuracy)
    F1Score = f1_score(yTarget_testData, prediction)
    print('F1Score :', F1Score)
    Con_Matrix = confusion_matrix(yTarget_testData, prediction)
    print(Con_Matrix)
    fpr, tpr, thresholds = roc_curve(yTarget_testData, prediction, pos_label=1)
    print('False Positive Rate:', fpr)
    print('True Positive Rate: ', tpr)
    #print('thresholds:', thresholds)
    AUC = auc(fpr, tpr)
    plt.figure(2)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show
    print(AUC)
    
    (words, prob) = DP.gettoptenwords(trainpos_probability_Dict)
    indexes = np.arange(len(words))
    width = .5
    plt.figure(4)
    plt.title('Positive Words Histogram Chart')
    plt.bar(indexes, prob, width)
    plt.xticks(indexes, words)
    plt.show()
    
    (wordsNeg, probNeg) = DP.gettoptenwords(trainneg_probability_Dict)
    index = np.arange(len(wordsNeg))
    wide = .5
    plt.figure(5)
    plt.title('Negative Words Histogram Chart')
    plt.bar(index, probNeg, wide)
    plt.xticks(indexes, wordsNeg)
    plt.show()
    
    
    
    
    
    
    
               
    
if __name__ == "__main__":
    main()
