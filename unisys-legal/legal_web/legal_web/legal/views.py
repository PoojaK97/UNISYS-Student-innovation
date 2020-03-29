from django.shortcuts import render
from django.http import HttpResponse
from django.http import Http404
from django.template import loader
import urllib.request as urllib2
import json
import sys
import math
#import settings
from django.conf import settings
import nltk.data
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.compat import python_2_unicode_compatible
import re
import operator
from gensim.summarization import summarize
import re
try:
    from django.contrib.staticfiles.templatetags.staticfiles import static
except ImportError:
    from django.templatetags.static import static
from django.contrib.staticfiles.storage import staticfiles_storage

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import requests

#NLTK specific imports
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tree import Tree



# Create your views here.
abilities = "Brief you about the document<br>List the dated events<br>Point the important words in it<br>"
defaultmsg = "Hi! Have anything to ask about the document? I can<br>" + abilities
rawstart = "Hi! Have anything to ask about the document/n/nI can brief you about the document, list the dated events, point the important words in it"
categories = ['Appellate Tribunal For Electricity','Central Administrative Tribunal','Central Information Commission','Competition Commission of India','Income Tax Appellate Tribunal','Consumer Disputes Redressal', 'National Green Tribunal' ,'Company Law Board', 'Customs, Excise and Gold Tribunal' , 'Securities Appellate Tribunal']
def testview(request):
   context = {
      'page_title' : 'Legal Case Studies',
      'readfile' : 'https://drive.google.com/file/d/1prRlrLEK6St6Lsp9O3FyUyc-A4a9KuUt/view'
   }
   template = loader.get_template('legal/startpage.html')
   return HttpResponse(template.render(context,request))

def readmeview(request):
   context = {
      'page_title' : 'Case Studies | Team DeepVaders',
      'readme' : 'https://drive.google.com/file/d/1prRlrLEK6St6Lsp9O3FyUyc-A4a9KuUt/view'
   }
   template = loader.get_template('legal/readme.html')
   return HttpResponse(template.render(context,request))


def responseview(request):
   if request.method != 'POST':
      raise Http404('Wrong url accessed. Access the main page or the readme')
   if 'myfile' not in request.FILES.keys():
      raise Http404('You have not uploaded the file. Please go to the main page and then upload a file in txt format')
   if request.FILES['myfile'].name.split('.')[1] != 'txt':
      raise Http404('The file is not in text format. Please upload the file in proper format')
   filetext = request.FILES['myfile'].read().decode('utf-8')
   #print (filetext)
   #read_legal_dict()
   legal_words = read_from_dict()
   orgs,persons,locs = loadorgspersonslocs(filetext)
   impwords = callapi(filetext)
   #print (impwords)
   name = request.FILES['myfile'].name.split('.')[0]
   print (name)
   #cs_summary = cs_summary(filetext,name)
   cs_preprocess = preprocess_cs(filetext)
   #df_vec = cal_df(cs_preprocess)
   #print (df_vec)
   #cs_summary = cal_tf_Idf(cs_preprocess, filetext)
   summary = prepsummary(filetext)
   chatsummary = prepchatsummary(filetext)
   #print (chatsummary)
   chatshortsummary = prepchatshortsummary(filetext)
   #print (chatshortsummary)
   #print ("hello")
   date_events = extractdates(filetext)
   #print ("date:" , date_events)
   real = preprocess(filetext,impwords)
   #print (real)
   sums = preprocess(summary,impwords)
   #print (sums)
   chatkeys = parsekeywords(impwords)
   #print (chatkeys)
   #print (settings.BASE_DIR)
   context = {
      'filename' : name,
      'realtext' : real,
      'summary' :  sums,
      'title' : 'Analysis of ' + name,
      'chatkeywords' : chatkeys,
      'chatsummary' : preprocess(chatsummary,impwords),
      'chatshortsummary' : preprocess(chatshortsummary,impwords),
      'defaultmsg' : defaultmsg,
      'rawstart' : rawstart,
      'rawchatsummary' : chatsummary,
      'rawchatshortsummary' : chatshortsummary,
      'rawchatkeywords' : impwords,
      'rawdates' : date_events,
      'chatdates' : date_events.replace('\n','<br>'),
      'category' : categories[1],
      'orgs' : orgs,
      'persons' : persons,
      'locs' : locs,
      #'cs_summary':cs_summary
   }
   #for index, value in context.items():
   #  print (index, ":" , value)
   template = loader.get_template('legal/index.html')
   return HttpResponse(template.render(context,request))

def prepsummary(filetext):
   sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.2)
   if len(sum) < 1:
      sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.3)
   if len(sum) < 1:
      sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.4)
   if len(sum) < 1:
      sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.5)
   if len(sum) < 1:
      sum = 'Sorry, the document is too small to be summarised'
   return sum

def prepchatsummary(filetext):
   sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.1)
   if len(sum) < 1:
      sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.2)
   if len(sum) < 1:
      sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.3)
   if len(sum) < 1:
      sum = 'Sorry, the document is too small to be summarised further'
   return sum

def prepchatshortsummary(filetext):
   sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.01)
   if len(sum) < 1:
      sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.05)
   if len(sum) < 1:
      sum = summarize(filetext,word_count=len(filetext.split(' ')) * 0.1)
   if len(sum) < 1:
      sum = 'Sorry, the document is too small to be summarised further'
   return sum

def callapi(filetext):
   data =  {
      "Inputs": {
            "input1":
            {
               "ColumnNames": ["1", "2", "Column 2"],
               "Values": [ [ "0", "0", filetext ], [ "0", "0", filetext ], ]
            },        },
            "GlobalParameters": {
      }
   }

   body = str.encode(json.dumps(data))

   url = 'https://ussouthcentral.services.azureml.net/workspaces/8423fe6354e64c5583076f21aa2f23c0/services/0c172734030249a9865bc7ba4e95351f/execute?api-version=2.0&details=true'
   api_key = '1I5Pdv19ADalbWoDfKO7/yhnGL4bgdymg0RUopI+cQGgh0P6/C0C8JB5kA0GKNudTj4tc4UAGrn5OQ15Pf1oiw=='
   headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
   req = urllib2.Request(url, body, headers) 

   try:
      response = urllib2.urlopen(req)

      result = json.loads(response.read())
      return result['Results']['output1']['value']['Values'][0][0] 
   except urllib2.HTTPError as error:
      raise Http404("The document failed to comprehend with status code: " + str(error.code))                 


def preprocess(filetext,impwords):
   catchphrases = impwords.split(',')
   data = filetext.replace('\n','<br>')
   for word in catchphrases:
      if len(word.split(' ')) > 2:
         data = data.replace(word,'<b><i>' + word + '</b></i>')
   return data

def parsekeywords(impwords):
   keyw = "<i>";
   ctf = impwords.split(',')
   for word in ctf:
      if len(word.split(' ')) > 2:
         keyw += word + "<br>"
   keyw += "</i>"
   return keyw

def extractdates(filetext):
   regex = r"([A-Z][^\.!?]*)(\d{1,2}[t][h]\s\D{3,8}[,]\s\d{2,4}|\d{0,1}[1][s][t]\s\D{3,8}[,]\s\d{2,4}|\d{0,1}[2][n][d]\s\D{3,8}[,]\s\d{2,4}|\d{0,1}[3][r][d]\s\D{3,8}[,]\s\d{2,4}|\d{1,2}\s\D{3,8}[,]\s\d{2,4})\s([a-z][^\.!?]*)([\.!?])"
   totalstr = ""
   matches = re.finditer(regex,filetext)
   for matchNum,match in enumerate(matches):
      totalstr += match.group()
   return totalstr


#change this URL before deploying
def get_category(filetext):
   r = requests.get(url = "http://localhost:8000/static/legal/naive.sav")
   #naiv=pickle.loads(r.content,encoding='latin1')
   #print (naiv)
   vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,n_features=2**16)
   categories=['aptels','cat','cic']
   r2 = requests.get(url = "http://localhost:8000/static/legal/eg.txt")
   test_data=[]
   test_data.append(filetext)
   test_data.append(r2.text)
   test = vectorizer.transform(test_data)
   #k = naiv.predict(test)
   return k[0]-1

def process_text(txt_file):
    raw_text = txt_file
    token_text = word_tokenize(raw_text)
    return token_text

def nltk_tagger(token_text):
    tagged_words = nltk.pos_tag(token_text)
    clean_tags = []
    for (i,j) in tagged_words:
        if(j=='NN'):
            clean_tags.append((i,j))
    ne_tagged = nltk.ne_chunk(tagged_words)
    return(ne_tagged)

def structure_ne(ne_tree):
    ne = []
    for subtree in ne_tree:
        if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne.append((ne_string, ne_label))
    return ne

def nltk_main(txt):
    return (structure_ne(nltk_tagger(process_text(txt))))

def get_tags(txt):
    ner_tags = nltk_main(txt)
    person = []
    orgs = []
    loc = []
    for (i,j) in ner_tags:
        if(j=='ORGANIZATION'):
            orgs.append(i)
        elif(j=='PERSON'):
            person.append(i)
        elif(j=='LOCATION' or j=='GPE'):
            loc.append(i)
    return (orgs,person,loc)

def clean_up(arr,c=20):
    if((len(arr)-len(set(arr)))/len(arr)>=0.8):
        return set(arr)
    else:
        freq = nltk.FreqDist(arr)
        arr_im = []
        if(c==0):
            for i in freq:
                if(freq[i]<=1):
                    if(len(i)>=8 and len(i)<=18):
                        arr_im.append(i)
            return set(arr_im)
        threshold = c * len(set(arr))/len(arr)
        for i in freq:
            if(freq[i]>=threshold):
                arr_im.append(i)
        return set(arr_im)

def run(txt):
    (o,p,l) = get_tags(txt)
    o = clean_up(o,c=20)
    p = clean_up(p,c=0)
    l = clean_up(l,c=5)
    return (list(o), list(p), list(l))

def loadorgspersonslocs(txt):
   orgstr = ""
   personstr = ""
   locstr = ""
   o,p,l = run(txt)
   for name in o:
      if name == 'Kanoon':
         pass
      else:
         orgstr = orgstr + name + ", "
   for name2 in l:
      if name2 == 'Kanoon':
         pass
      else:
         locstr = locstr + name2 + ", "
   for name3 in p:
      if name3 == 'Kanoon':
         pass
      else:
         personstr = personstr + name3 + ", "
   
   orgstr = orgstr[:-2]
   locstr = locstr[:-2]
   personstr = personstr[:-2]
   return orgstr,personstr,locstr

def preprocess_cs(filetext):
    #fileName = os.path.join(input_path,f)
    #file = open(input_path, 'r')

    documents = []
    documents = (filetext).strip().split("\n");
    #file.close()

    for i in range(0,len(documents)):
        line = documents[i]
        if line.startswith("1."):
            break
    doc = documents[i:]
    temp = ""
    for eachDocument in doc[:]:
        eachDocument = re.sub(r'(\d\d\d|\d\d|\d)\.\s', ' ', eachDocument)#removes the paragraph lables 1. or 2. etc.
        eachDocument = re.sub(r'(?<=[a-zA-Z])\.(?=\d)', '', eachDocument)#removes dot(.) i.e File No.1063
        eachDocument = re.sub(r'(?<=\d|[a-zA-Z])\.(?=\s[\da-z])', ' ', eachDocument)#to remove the ending dot of abbr
        eachDocument = re.sub(r'(?<=\d|[a-zA-Z])\.(?=\s?[\!\"\#\$\%\&\'\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~])', '', eachDocument)#to remove the ending dot of abbr
        eachDocument = re.sub(r'(?<!\.)[\!\"\#\$\%\&\'\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', ' ', eachDocument)#removes the other punctuations

        temp = temp +''+eachDocument
    documents = []
    temp = temp.replace("  "," ")
    documents = temp.replace(" ","",1)
    return documents


def get_continuous_chunks(text):
    #print (text)

    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    #print (chunked)

    #print ("ne")
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    return continuous_chunk

def read_from_dict():
   legal_words=[]
   f_l = open(os.path.join(settings.BASE_DIR, 'dictionary.txt'), 'r')
   for wd in f_l:
      legal_words.append(wd)
   f_l.close()
   #print (legal_words)
   return legal_words

def cal_df(preprocess):
   df_vec={}
   tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
   data = preprocess
   sntncs = tokenizer.tokenize(data)
   nor_stp_lmt = []
   try:
      wordnet_lemmatizer = WordNetLemmatizer()
      stop = set(stopwords.words('english'))
      for s in sntncs:
         s_nor_stp_lmt = ""
         s = s.lower()
         words = word_tokenize(s)
         for w in words:
            if w not in stop:
               w = wordnet_lemmatizer.lemmatize(w)
               s_nor_stp_lmt = s_nor_stp_lmt + w + " "
         nor_stp_lmt.append(s_nor_stp_lmt)
   except:
      print ("wordnet")
   unq_words = {}
   for s in nor_stp_lmt:
      for w in word_tokenize(s):
         if w != ".":
            if w not in unq_words:
               unq_words[w] = 0
   for k in unq_words.keys():
      if k in df_vec:
         df_vec[k] = df_vec[k] + 1
      else:
         df_vec[k] = 1
   return df_vec

def cal_tf_Idf(preprocess, input_path):
   legal_words = read_from_dict()
   total_docs = 1
   sumr = ""
   doc_w_vec = {}
   df_vec = cal_df(preprocess)
   #sums = []
   tf_idf_sntnc = {}
   tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
   data = preprocess
   sntncs = tokenizer.tokenize(data)
   nor_stp_lmt = []
   stp_lmt_cased = []
   wordnet_lemmatizer = WordNetLemmatizer()
   stop = set(stopwords.words('english'))
   for s in sntncs:
      s_nor_stp_lmt = ""
      s_u = s.lower()
      words = word_tokenize(s_u)
      for w in words:
         if w not in stop:
            w = wordnet_lemmatizer.lemmatize(w)
            s_nor_stp_lmt = s_nor_stp_lmt + w + " "
      nor_stp_lmt.append(s_nor_stp_lmt)
      words = word_tokenize(s)
      case_sntnc = ""
      for w in words:
         if w not in stop:
            w = wordnet_lemmatizer.lemmatize(w)
            case_sntnc = case_sntnc + w + " "
      stp_lmt_cased.append(case_sntnc)
      tf_vec = {}
      length = 0
      for i in range(len(nor_stp_lmt)):
         s = nor_stp_lmt[i]
         for w in word_tokenize(s):
            if w != ".":
               length = length + 1
               if w in tf_vec:
                  tf_vec[w] = tf_vec[w] + 1
               else:
                  tf_vec[w] = 1
      tf_idf_doc = {}
      for k in tf_vec.keys():
         tf_vec[k] = float(tf_vec[k])/float(length)
         tf_idf_doc[k] = tf_vec[k] * math.log10(float(total_docs)/float(df_vec[k]))

      doc_w_vec[input_path] = tf_idf_doc
      std_list = []
      try:
         for i in range(len(nor_stp_lmt)):
            s = nor_stp_lmt[i]
            ac_s = sntncs[i]
            sm = 0
            no_of_words = len(word_tokenize(s))
            for w in word_tokenize(s):
               if w in tf_idf_doc.keys():
                  sm = sm + tf_idf_doc[w]
            tf_idf_s = float(sm)/float(no_of_words)
            tf_idf_sntnc[ac_s] = tf_idf_s
            std_list.append(tf_idf_s)
      except:
         print("this it")
      sd = np.std(std_list)
      for i in range(len(nor_stp_lmt)):
         cased_s = stp_lmt_cased[i]
         ne_list = get_continuous_chunks(cased_s)
         ac_s = sntncs[i]
         e = float(len(ne_list))/float(len(word_tokenize(nor_stp_lmt[i])))
         op = any(char.isdigit() for char in s)
         d = 0
         if op:
            d = 1
         words = word_tokenize(nor_stp_lmt[i])
         bag = []
         for wd in words:
            try:
               wd = wd.replace("[","").replace("]","").replace("(","").replace(")","").replace("{","").replace("}","")
               r = re.compile(wd + ".*")
            except:
               print ('efrr1')
            newlist = list(filter(r.match, legal_words))
            for item in newlist:
               if item in nor_stp_lmt[i]:
                  bag.extend(item.split(" "))
         myset = set(bag)
         g = float(len(myset))/float(len(words))
         tf_idf_sntnc[ac_s] = tf_idf_sntnc[ac_s] + sd*(0.2 * d + 0.3 * e + 1.5 * g)
      sorted_x = sorted(tf_idf_sntnc.items(), key=operator.itemgetter(1), reverse=True)
      print (sorted_x)
      for pair in sorted_x:
         sumr = sumr + pair[0] + " "
   #print (sumr)
   sumr = sumr.split('.')
   sumr_new=""
   for i in range (math.floor(len(sumr)*0.3)):
      sumr_new += sumr[i] + '.'
   print (sumr_new)
   return (sumr_new)