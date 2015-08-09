import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import glob
import numpy as np
import sys
import Image
import re

dictionaryLoc = "/home/testing32/Downloads/shopping/images/"
twoClass = {'bag':0,'shoe':1}
fourClass = {'bags_clutch':0, 'bags_hobo':1, 'womens_flats':2, 'women_pumps':4}
    
SIM_LIM = 10
STOP_WORDS = ("\n", "\t", "closeouts",
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because",
    "been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does",
    "doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have",
    "haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't",
    "my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over",
    "own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the",
    "their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this",
    "those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't",
    "what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would",
    "wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves")

class ImageClass(object):
    
    def __init__(self, ssd, fileLoc):
        self.ssd = ssd
        self.fileLoc = fileLoc
        
    def __cmp__(self, other):
        return cmp(self.ssd, other.ssd)


class TextClass(object):

    def __init__(self, score, fileLoc):
        self.score = score
        self.fileLoc = fileLoc
        
    def __cmp__(self, other):
        return cmp(self.score, other.score)
    

class CombinedClass(object):
    
    def __init__(self, imageSSD, textSSD, fileLoc,alpha=.000002):
        self.score = alpha*imageSSD + (1.0 - alpha) * textSSD
        self.fileLoc = fileLoc
        
    def __cmp__(self, other):
        return cmp(self.score, other.score)
    
     
    
def deepLearningQueryPrediction(queryImage, dictionaryDir):
    # uses the convolution NN generated from myconvnet to predict
    # labels of query images
    import cPickle
    import itertools
      
    def getTwoClassName(prediction):
        if prediction==twoClass['bag']:
            return "bags"
        else:
            return "shoes"
    
    def getFourClassName(prediction):
        if prediction==fourClass['bags_clutch']:
            return "bags_clutch"
        elif prediction==fourClass['bags_hobo']:
            return "bags_hobo"
        elif prediction==fourClass['womens_flats']:
            return "womens_flats"
        else:
            return "women_pumps"
        
    queryImageLoc = "/home/testing32/Downloads/shopping/queryimages/"
    
    # gets a base list of dictionary files
    fileNames = [re.search('descr_(.+?)\.txt', s.lower()).group(1) for s in glob.glob(queryImageLoc + "descr_*.txt")]
    
    predict = cPickle.load(open("bestModel_99_2class.mdl","rb"))
    
    for filename in fileNames:
        imageName = queryImageLoc + "img_" + filename + ".jpg"
        image = np.array(list(itertools.chain.from_iterable(np.array(loadImage(imageName))))).reshape(-1, 1, 32, 32)
        #print filename + "\t\tpredicted to be: " + getFourClassName(predict(image)[0])
        print filename + "\t\tpredicted to be: " + getTwoClassName(predict(image)[0])
    

    
# four class
def parseFilenameToTwoClass(filename):
    if re.match(r"bags*",filename):
        return twoClass['bag']
    else: # women pumps
        return twoClass['shoe']

# TWO CLASS
def parseFilenameToFourClass(filename):
    if re.match(r"bags_clutch*",filename):
        return fourClass["bags_clutch"]
    elif re.match(r"bags_hobo*",filename):
        return fourClass["bags_hobo"]
    elif re.match(r"womens_flats*",filename):
        return fourClass["womens_flats"]
    else: # women pumps
        return fourClass["womens_pumps"]
      

def evaluateResults(queryFilename, results):
    return 1

def loadImage(fullImageLocation):
    return Image.open(fullImageLocation).convert("L").resize((32,32), Image.ANTIALIAS)

def loadText(fullTextLocation, idfDict=None):
    queryDescr = open(fullTextLocation, "r")
    text = queryDescr.readlines()

    cleanQueryText = cleanText(text[0])
    return dictionaryList(cleanQueryText, idfDict=idfDict)

def cleanText(text):
    # removes stop words from a string and returns a list of the remaining words
    return [s.lower() for s in re.findall(r"[A-Za-z']+",text) if s.lower() not in STOP_WORDS]
  
def dictionaryList(wordList, idfDict=None):
    wordDict = {}
    
    if len(wordList)==0:
        return wordDict
    
    incrementValue = 1/float(len(wordList))
    
    for word in wordList:
        if wordDict.has_key(word):
            wordDict[word] += incrementValue
        else:
            wordDict[word] = incrementValue
    
    if idfDict != None:
        total = 0.0
        for key in wordDict.keys():
            wordDict[key] = (.5 + .5 * wordDict[key])
            if idfDict.has_key(key):
                wordDict[key] = wordDict[key] * idfDict[key]
            total += wordDict[key]
            
        # normalize
        for key in wordDict.keys():
            wordDict[key] = wordDict[key] / total
    
    return wordDict


def getIDFDict(fileNames):
    weights = {}
    
    for filename in fileNames:
        queryDescr = open(filename, "r")
        text = queryDescr.readlines()

        cleanQueryText = cleanText(text[0])
        
        for word in list(set(cleanQueryText)):
            if weights.has_key(word):
                weights[word] += 1
            else:
                weights[word] = 1
                
    for key in weights.keys():
        weights[key] = math.log(len(fileNames)/weights[key])
                
    return weights


def grayScale(img):
    gray = np.zeros((img.shape[0], img.shape[1])) # init 2D numpy array
    
    # get row number
    for rownum in range(len(img)):
        for colnum in range(len(img[rownum])):
            gray[rownum][colnum] = np.average(img[rownum][colnum])
          
    return gray
    
def imageSSD(imgOne, imgTwo):
    return np.sum((imgOne - imgTwo) ** 2)
  
def textSSD(dictOne, dictTwo):
    score = 0
    if not dictOne or not dictTwo:
        return sys.float_info.max
    
    for key in dictOne:
        if dictTwo.has_key(key):
            score += (dictOne[key] - dictTwo[key])**2
        else:
            score += dictOne[key] ** 2
    
    for key in dictTwo:
        if not dictOne.has_key(key):
            score += dictTwo[key] ** 2
    
    return score
  
def getTextList(queryDict, fileNames):
    # for each file, load the text SSD and 
    # add them to the list of text objects 
    textList = []
    for filename in fileNames:
        textList.append(TextClass(
          textSSD(queryDict, loadText(filename)),
          filename))
        
    textList.sort()
    return textList


def getImageList(queryImage, fileNames):
    imageList = []
    
    # for each file, load the image SSD and 
    # add them to the list of image objects 
    for filename in fileNames:
        imageList.append(ImageClass(
          imageSSD(np.asanyarray(queryImage), np.asanyarray(loadImage(filename))),
          filename))
        
    imageList.sort()
    return imageList

    
def queryImage(queryImage, dictionaryDir):
    # Queries the dictionary based on a query image
    
    # Gets a list of all images with their SSD
    imageList = getImageList(loadImage(queryImage), glob.glob(dictionaryDir + "*.jpg"))
    """
    img = mpimg.imread(queryImage)
    plt.imshow(img, origin='lower')
    plt.show()
    """
    
    # print the top 10 results
    print "Image Results for " + queryImage
    count = 1
    for imageItem in imageList[:10]:
        print str(count) + ": score=" + str(imageItem.ssd) + " : " + imageItem.fileLoc
        count += 1
        """
        img = mpimg.imread(imageItem.fileLoc)
        plt.imshow(img, origin='lower')
        plt.show()"""
        
    generateHTMLFile(imageList[:10])

def queryDescription(queryText, dictionaryDir):
    # Queries dictionary based on text
    
    # gets the SSD of all dictionary descriptions given a query description
    textList = getTextList(loadText(queryText), glob.glob(dictionaryDir + "descr_*.txt"))
    
    # print the top 10 results
    print "Text Results for " + queryText
    count = 1
    for textItem in textList[:10]:
        print str(count) + ": score=" + str(textItem.score) + " : " + textItem.fileLoc
        count += 1

    generateHTMLFile(textList[:10])
    

def queryCombined(queryImageName, queryTextName, dictionaryDir):
    # Queries both image and description to get a list of matches
    
    # gets a base list of dictionary files
    fileNames = [re.search('descr_(.+?)\.txt', s.lower()).group(1) for s in glob.glob(dictionaryDir + "descr_*.txt")]
    
    # gets a list of objects which have the combined SSD of image and text
    combinedList = []
    
    queryImage = loadImage(queryImageName)
    queryText = loadText(queryTextName)
    
    # for each file name, load the image SSD, text SSD and 
    # add them to the list of combined objects 
    for filename in fileNames:
        fileImageSSD = imageSSD(np.asanyarray(queryImage), np.asanyarray(loadImage(dictionaryDir + "img_" + filename + ".jpg")))
        fileTextSSD = textSSD(queryText, loadText(dictionaryDir + "descr_" + filename + ".txt"))
        combinedList.append(CombinedClass(fileImageSSD, fileTextSSD, dictionaryDir + "img_" + filename + ".jpg"))
        
    combinedList.sort()
    # print the top 10 ranking
    print "Combined Results for " + queryImageName
    count = 1
    for combinedItem in combinedList[:10]:
        print str(count) + ": score=" + str(combinedItem.score) + " : " + combinedItem.fileLoc
        count += 1
    
    generateHTMLFile(combinedList[:10])
    
def testTextQuery(queryText, dictionaryDir):
    # Queries dictionary based on text
    
    # gets the SSD of all dictionary descriptions given a query description
    fileNames = glob.glob(dictionaryDir + "descr_*.txt")
    
    idfDict = getIDFDict(fileNames)
    queryDict = loadText(queryText, idfDict=idfDict)
    
    # for each file, load the text SSD and 
    # add them to the list of text objects 
    textList = []
    for filename in fileNames:
        textList.append(TextClass(
          textSSD(queryDict, loadText(filename, idfDict=idfDict)),
          filename))
        
    textList.sort()

    # print the top 10 results
    print "Text Results for " + queryText
    count = 1
    for textItem in textList[:10]:
        print str(count) + ": score=" + str(textItem.score) + " : " + textItem.fileLoc
        count += 1
    
def partFour(queryImageName, queryTextName, dictionaryDir, twoClassModel, fourClassModel, displayImages=False):
    # Queries both image and description to get a list of matches
    # while using my own ideas to try and get better results
    import cPickle
    import itertools
    import cv2
    
    scalingConstant = 0.000021556
    imageAlpha = .6
    
    twoClassPredictor = cPickle.load(open(twoClassModel,"rb"))
    fourClassPredictor = cPickle.load(open(fourClassModel,"rb"))
    
    idfDict = getIDFDict(glob.glob(dictionaryDir + "descr_*.txt"))
    
    # gets a base list of dictionary files
    fileNames = [re.search('descr_(.+?)\.txt', s.lower()).group(1) for s in glob.glob(dictionaryDir + "descr_*.txt")]
    
    # gets a list of objects which have the combined SSD of image and text
    queryImage = loadImage(queryImageName)
    
    imagePrediction = np.array(list(itertools.chain.from_iterable(np.array(queryImage)))).reshape(-1, 1, 32, 32)
    twoClassPrediction = twoClassPredictor(imagePrediction)[0]
    fourClassPrediction = fourClassPredictor(imagePrediction)[0]
    
    queryHist = cv2.calcHist([cv2.imread(queryImageName,0)],[0],None,[256],[0,256])
    queryHist = cv2.normalize(queryHist).flatten()
    
    queryText = loadText(queryTextName, idfDict=idfDict)
    
    combinedList = []
    
    if displayImages:
        img = mpimg.imread(queryImageName)
        plt.imshow(img, origin='lower')
        plt.show()
    
    # for each file name, load the image SSD, text SSD and 
    # add them to the list of combined objects 
    for filename in fileNames:
        # if our model says that this filename isn't in our predicted shoe/bag
        # category then we don't consider it
        if parseFilenameToTwoClass(filename) != twoClassPrediction:
            continue
        
        imageName = dictionaryDir + "img_" + filename + ".jpg"
        
        hist = cv2.calcHist([cv2.imread(imageName,0)],[0],None,[256],[0,256])
        hist = cv2.normalize(hist).flatten()
        
        # intersect is giving the best results so we will stick with that one
        histCompareResult = 1/cv2.compareHist(queryHist, hist, cv2.cv.CV_COMP_INTERSECT)
        
        fileImageSSD = scalingConstant*imageSSD(np.asanyarray(queryImage), np.asanyarray(loadImage(imageName)))
        fileImageSSD = fileImageSSD * (imageAlpha) + histCompareResult * (1 - imageAlpha)
        
        fileTextSSD = textSSD(queryText, loadText(dictionaryDir + "descr_" + filename + ".txt", idfDict=idfDict))
        
        # we give some added weight if we our model thinks 
        # this image is the same category as the query image
        if fourClassPrediction==parseFilenameToFourClass:
            fileImageSSD = fileImageSSD * .982
        
        combinedList.append(CombinedClass(fileImageSSD, fileTextSSD, imageName, alpha=.3))
        
    combinedList.sort()

    # print the top 10 ranking
    print "Part Four Results for " + queryImageName
    count = 1
    for combinedItem in combinedList[:10]:
        print str(count) + ": score=" + str(combinedItem.score) + " : " + combinedItem.fileLoc
        count += 1
        if displayImages:
            img = mpimg.imread(combinedItem.fileLoc)
            plt.imshow(img, origin='lower')
            plt.show()
            
    generateHTMLFile(combinedList[:30])
    
def histogram(queryImage, dictionaryDir):
    import cv2
    
    img = cv2.imread(queryImage,0)
    queryHist = cv2.calcHist([img],[0],None,[256],[0,256])
    queryHist = cv2.normalize(queryHist).flatten()
    
    # gets a base list of dictionary files
    fileNames = [re.search('descr_(.+?)\.txt', s.lower()).group(1) for s in glob.glob(dictionaryDir + "descr_*.txt")]
    imageList = []
    
    for filename in fileNames:
        imageName = dictionaryDir + "img_" + filename + ".jpg"
        
        # hard coded class, make sure to change this if you are
        # searching for different classes
        if parseFilenameToTwoClass(filename) != 1:
            continue
        
        img = cv2.imread(imageName,0)
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        hist = cv2.normalize(hist).flatten()
        
        # intersect is giving the best results so we will stick with that one
        result = 1/cv2.compareHist(queryHist, hist, cv2.cv.CV_COMP_INTERSECT)
        #result = 1/cv2.compareHist(queryHist, hist, cv2.cv.CV_COMP_CORREL)
        #result = cv2.compareHist(queryHist, hist, cv2.cv.CV_COMP_CHISQR)
        #result = cv2.compareHist(queryHist, hist, cv2.cv.CV_COMP_BHATTACHARYYA)
        
        imageList.append(ImageClass(result, imageName))
    
    imageList.sort()
    """
    img = mpimg.imread(queryImage)
    plt.imshow(img, origin='lower')
    plt.show()
    """
    # print the top 10 results
    print "Image Results for " + queryImage
    count = 1
    for imageItem in imageList[:10]:
        print str(count) + ": score=" + str(imageItem.ssd) + " : " + imageItem.fileLoc
        count += 1
        """
        img = mpimg.imread(imageItem.fileLoc)
        plt.imshow(img, origin='lower')
        plt.show()"""
   
def generateHTMLFile(itemList):
    # generates an html files which contains the images
    # this made it easy to add pictures to the hw document
    html = "<html><body>"
    
    for item in itemList:
        html += '<img src="' + item.fileLoc + '" />'
    
    html += "</body></html>"
    myHtmlDoc = open("result.html", 'w')
    myHtmlDoc.write(html)
    myHtmlDoc.close()

if __name__ == "__main__":
    #queryImage("/home/testing32/Downloads/shopping/queryimages/img_womens_flats_800.jpg", dictionaryLoc)
    #queryImage("/home/testing32/Downloads/shopping/queryimages/img_bags_hobo_500.jpg", dictionaryLoc)
    
    #queryDescription("/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_300.txt", dictionaryLoc)
    #queryDescription("/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_1.txt", dictionaryLoc)
    
    #queryCombined("/home/testing32/Downloads/shopping/queryimages/img_bags_hobo_500.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_bags_hobo_500.txt", dictionaryLoc)
    #queryCombined("/home/testing32/Downloads/shopping/queryimages/img_womens_pumps_1.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_1.txt", dictionaryLoc)
    #queryCombined("/home/testing32/Downloads/shopping/queryimages/img_womens_flats_800.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_womens_flats_800.txt", dictionaryLoc)
    #queryCombined("/home/testing32/Downloads/shopping/queryimages/img_womens_pumps_300.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_300.txt", dictionaryLoc)
    #queryCombined("/home/testing32/Downloads/shopping/queryimages/img_bags_hobo_500.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_1.txt", dictionaryLoc)
    
    #partFour("/home/testing32/Downloads/shopping/queryimages/img_bags_clutch_200.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_bags_clutch_200.txt", dictionaryLoc, "bestModel_99_2class.mdl", "bestModel_98_4class.mdl")
    #partFour("/home/testing32/Downloads/shopping/queryimages/img_bags_hobo_500.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_bags_hobo_500.txt", dictionaryLoc, "bestModel_99_2class.mdl", "bestModel_98_4class.mdl")
    #partFour("/home/testing32/Downloads/shopping/queryimages/img_womens_pumps_1.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_1.txt", dictionaryLoc, "bestModel_99_2class.mdl", "bestModel_98_4class.mdl")
    partFour("/home/testing32/Downloads/shopping/queryimages/img_womens_flats_800.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_womens_flats_800.txt", dictionaryLoc, "bestModel_99_2class.mdl", "bestModel_98_4class.mdl")
    #partFour("/home/testing32/Downloads/shopping/queryimages/img_womens_pumps_300.jpg", "/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_300.txt", dictionaryLoc, "bestModel_99_2class.mdl", "bestModel_98_4class.mdl")
    
    #deepLearningQueryPrediction("/home/testing32/Downloads/shopping/queryimages/img_bags_clutch_1.jpg", dictionaryLoc)
    #testTextQuery("/home/testing32/Downloads/shopping/queryimages/descr_womens_pumps_300.txt", dictionaryLoc)
    
    #histogram("/home/testing32/Downloads/shopping/queryimages/img_bags_clutch_200.jpg", dictionaryLoc)
    #histogram("/home/testing32/Downloads/shopping/queryimages/img_bags_hobo_500.jpg", dictionaryLoc)
    #histogram("/home/testing32/Downloads/shopping/queryimages/img_womens_pumps_1.jpg", dictionaryLoc)
    #histogram("/home/testing32/Downloads/shopping/queryimages/img_womens_flats_800.jpg", dictionaryLoc)
    #histogram("/home/testing32/Downloads/shopping/queryimages/img_womens_pumps_300.jpg", dictionaryLoc)
