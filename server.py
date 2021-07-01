#!/bin/env  python3
from __future__ import print_function
from flask import Flask, render_template, request, session, Response, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask import jsonify

import json
import shutil
import bcrypt
import hashlib
import tempfile
import random
import string
import re
import pytz
import os
from os import listdir
import pandas as pd
import numpy as np
from numpy import array

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from collections import Counter
from datetime import datetime
from more_functions import *

import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import pickle

app=Flask(__name__)
#datadir="/export/ratspub/"
#datadir = "."
datadir="./"

app.config['SECRET_KEY'] = '#DtfrL98G5t1dC*4'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+datadir+'userspub.sqlite'
db = SQLAlchemy(app)
nltk.data.path.append("./nlp/")

# Sqlite database
class users(db.Model):
    __tablename__='user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

# Preprocessing of words for CNN
def clean_doc(doc, vocab):
    doc = doc.lower()
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))    
    tokens = [re_punc.sub('' , w) for w in tokens]    
    tokens = [word for word in tokens if len(word) > 1]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    return tokens

# Load tokenizer
with open('./nlp/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load vocabulary
with open('./nlp/vocabulary.txt', 'r') as vocab:
    vocab = vocab.read()

def tf_auc_score(y_true, y_pred):
    return tensorflow.metrics.auc(y_true, y_pred)[1]

K.clear_session()

# Create the CNN model
def create_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_length))
    model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = tensorflow.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf_auc_score])
    return model

# Use addiction ontology by default
onto_cont=open("addiction.onto","r").read()
dictionary=ast.literal_eval(onto_cont)


@app.route("/")
def root():
    if 'email' in session:
        ontoarchive()
        onto_len_dir = session['onto_len_dir']
        onto_list = session['onto_list']
    else: 
        onto_len_dir = 0
        onto_list = ''

    onto_cont=open("addiction.onto","r").read()
    dict_onto=ast.literal_eval(onto_cont)
    return render_template('index.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)


@app.route("/login", methods=["POST", "GET"])
def login():
    onto_len_dir = 0
    onto_list = ''
    onto_cont=open("addiction.onto","r").read()
    dict_onto=ast.literal_eval(onto_cont)
    email = None

    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        found_user = users.query.filter_by(email=email).first()
        if (found_user and (bcrypt.checkpw(password.encode('utf8'), found_user.password))):
            session['email'] = found_user.email
            #print(bcrypt.hashpw(session['email'].encode('utf8'), bcrypt.gensalt()))
            session['hashed_email'] = hashlib.md5(session['email'] .encode('utf-8')).hexdigest()
            session['name'] = found_user.name
            session['id'] = found_user.id
            flash("Login Succesful!")
            ontoarchive()
            onto_len_dir = session['onto_len_dir']
            onto_list = session['onto_list']
        else:
            flash("Invalid username or password!", "inval")
            return render_template('signup.html')
    return render_template('index.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)


@app.route("/signup", methods=["POST", "GET"])
def signup():
    onto_len_dir = 0
    onto_list = ''
    onto_cont=open("addiction.onto","r").read()
    dict_onto=ast.literal_eval(onto_cont)

    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        found_user = users.query.filter_by(email=email).first()

        if (found_user and (bcrypt.checkpw(password.encode('utf8'), found_user.password)==False)):
            flash("Already registered, but wrong password!", "inval")
            return render_template('signup.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)  

        session['email'] = email
        session['hashed_email'] = hashlib.md5(session['email'] .encode('utf-8')).hexdigest()
        session['name'] = name
        password = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())
        user = users(name=name, email=email, password = password)       
        if found_user:
            session['email'] = found_user.email
            session['hashed_email'] = hashlib.md5(session['email'] .encode('utf-8')).hexdigest()
            session['id'] = found_user.id
            found_user.name = name
            db.session.commit()
            ontoarchive()
            onto_len_dir = session['onto_len_dir']
            onto_list = session['onto_list']
        else:
            db.session.add(user)
            db.session.commit()
            newuser = users.query.filter_by(email=session['email']).first()
            session['id'] = newuser.id
            os.makedirs(datadir+"/user/"+str(session['hashed_email']))
            session['user_folder'] = datadir+"/user/"+str(session['hashed_email'])
            os.makedirs(session['user_folder']+"/ontology/")

        flash("Login Succesful!")
        return render_template('index.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)
    else:
        if 'email' in session:
            flash("Already Logged In!")
            return render_template('index.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)
        return render_template('signup.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)


@app.route("/signin", methods=["POST", "GET"])
def signin():
    email = None
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        found_user = users.query.filter_by(email=email).first()

        if (found_user and (bcrypt.checkpw(password.encode('utf8'), found_user.password))):
            session['email'] = found_user.email
            session['hashed_email'] = hashlib.md5(session['email'].encode('utf-8')).hexdigest()
            session['name'] = found_user.name
            session['id'] = found_user.id
            flash("Login Succesful!")
            #onto_len_dir = 0
            #onto_list = ''
            onto_cont=open("addiction.onto","r").read()
            ontoarchive()
            onto_len_dir = session['onto_len_dir']
            onto_list = session['onto_list']
            dict_onto=ast.literal_eval(onto_cont)
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)
        else:
            flash("Invalid username or password!", "inval")
            return render_template('signup.html')   
    return render_template('signin.html')

# change password 
@app.route("/<nm_passwd>", methods=["POST", "GET"])
def profile(nm_passwd):
    try:
        if "_" in str(nm_passwd):
            user_name = str(nm_passwd).split("_")[0]
            user_passwd = str(nm_passwd).split("_")[1]
            user_passwd = "b\'"+user_passwd+"\'"
            found_user = users.query.filter_by(name=user_name).first()

            if request.method == "POST":
                password = request.form['password']
                session['email'] = found_user.email
                session['hashed_email'] = hashlib.md5(session['email'] .encode('utf-8')).hexdigest()
                session['name'] = found_user.name
                session['id'] = found_user.id
                password = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())
                found_user.password = password
                db.session.commit()
                flash("Your password is changed!", "inval")
                onto_len_dir = 0
                onto_list = ''
                onto_cont=open("addiction.onto","r").read()
                dict_onto=ast.literal_eval(onto_cont)
                return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)
            # remove reserved characters from the hashed passwords
            reserved = (";", "/", "?", ":", "@", "=", "&", ".")
            def replace_reserved(fullstring):
                for replace_str in reserved:
                    fullstring = fullstring.replace(replace_str,"")
                return fullstring
            replaced_passwd = replace_reserved(str(found_user.password))

            if replaced_passwd == user_passwd:
                return render_template("/passwd_change.html", name=user_name)
            else:
                return "This url does not exist"
        else: 
            return "This url does not exist"
    except (AttributeError):
        return "This url does not exist"


@app.route("/logout")
def logout():
    onto_len_dir = 0
    onto_list = ''
    onto_cont=open("addiction.onto","r").read()
    dict_onto=ast.literal_eval(onto_cont)

    if 'email' in session:
        global user1
        if session['name'] != '':
            user1 = session['name']
        else: 
            user1 = session['email']
    flash("You have been logged out, {user1}", "inval")
    session.pop('email', None)
    session.clear()
    return render_template('index.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)


@app.route("/about")
def about():
    return render_template('about.html')


# Ontology selection
@app.route("/index_ontology", methods=["POST", "GET"])
def index_ontology():
    namecat2 = request.args.get('onto')
    session['namecat']=namecat2

    if (namecat2 == 'addiction' or namecat2 == 'Select your ontology' ):
        session['namecat']='addiction'
        onto_cont=open("addiction.onto","r").read()
    else:
        dirlist = os.listdir(session['user_folder']+"/ontology/")
        for filename in dirlist:
            onto_name = filename.split('_0_')[1]
            if namecat2 == onto_name:
                onto_cont = open(session['user_folder']+"/ontology/"+filename+"/"+namecat2+".onto", "r").read()
                break
    dict_onto=ast.literal_eval(onto_cont)
    onto_len_dir = session['onto_len_dir']
    onto_list = session['onto_list']
    return render_template('index.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = session['namecat'], dict_onto=dict_onto )


@app.route("/ontology", methods=["POST", "GET"])
def ontology():
    namecat2 = request.args.get('onto')
    select_date=request.args.get('selected_date')
    namecat_exist=0

    if select_date != None:
        time_extension = str(select_date)
        time_extension = time_extension.split('_0_')[0]
        namecat = str(select_date).split('_0_')[1]
        time_extension = time_extension.replace(':', '_')
        time_extension = time_extension.replace('-', '_')

        if ('email' in session):
            session['namecat'] = session['user_folder']+"/ontology/"+str(time_extension)+"_0_"+namecat+"/"+namecat
        else:
            session['namecat']=tempfile.gettempdir()+'/'+namecat
        onto_cont = open(session['namecat']+".onto","r").read()

        if onto_cont=='':
            dict_onto={}
        else:
            dict_onto=ast.literal_eval(onto_cont)
    elif (('email' in session) and (namecat2 == 'addiction')):
        namecat='addiction'
        session['namecat']=namecat
        onto_cont = open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
    else:
        if (('email' in session) and ((namecat2 != None) and (namecat2 != 'choose your ontology'))):
            namecat=namecat2
            dirlist = os.listdir(session['user_folder']+"/ontology/")

            for filename in dirlist:
                onto_name = filename.split('_0_')[1]
                if onto_name==namecat:
                    namecat_exist=1
                    namecat_filename=filename
                    break
            if namecat_exist==1:
                session['namecat'] = session['user_folder']+"/ontology/"+namecat_filename+'/'+namecat
            else:
                onto_cont = open("addiction.onto","r").read()
                dict_onto=ast.literal_eval(onto_cont)
        else:
            namecat='addiction'
            session['namecat']=namecat
            onto_cont = open("addiction.onto","r").read()
            dict_onto=ast.literal_eval(onto_cont)

    if request.method == "POST":
        maincat = request.form['maincat']
        subcat = request.form['subcat']
        keycat = request.form['keycat']
        namecat = request.form['namecat']
        namecat_exist=0
        maincat=re.sub('[^,a-zA-Z0-9 \n]', '', maincat)
        subcat=re.sub('[^,a-zA-Z0-9 \n]', '', subcat)
        keycat=re.sub('[^,a-zA-Z0-9 \n]', '', keycat)
        keycat = keycat.replace(',', '|')
        keycat = re.sub("\s+", ' ', keycat)
        keycat = keycat.replace(' |', '|')
        keycat = keycat.replace('| ', '|')
        namecat=re.sub('[^,a-zA-Z0-9 \n]', '', namecat)

        # Generate a unique session ID depending on timestamp to track the results 
        timestamp = datetime.utcnow().replace(microsecond=0)
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = timestamp.astimezone(pytz.timezone("America/Chicago"))
        timeextension = str(timestamp)
        timeextension = timeextension.replace(':', '_')
        timeextension = timeextension.replace('-', '_')
        timeextension = timeextension.replace(' ', '_')
        timeextension = timeextension.replace('_06_00', '')
        session['timeextension'] = timeextension

        if request.form['submit'] == 'add':  # Add new keywords or create a new ontology
            if ('email' in session):
                session['namecat']=namecat
                if (namecat=='addiction'):
                    flash("You cannot change addiction keywords but a new ontology will be saved as 'addictionnew', instead","inval")
                    namecat='addictionnew'
                    session['namecat']=namecat
                session['user_folder'] = datadir+"/user/"+str(session['hashed_email'])
                dirlist = os.listdir(session['user_folder']+"/ontology/")
                for filename in dirlist:
                    onto_name = filename.split('_0_')[1]
                    if onto_name==namecat:
                        namecat_exist=1  # Add new keywords
                        namecat_filename=filename
                        break
                if namecat_exist==0:  # Create a new ontology folder
                    os.makedirs(session['user_folder']+"/ontology/"+str(timeextension)+"_0_"+namecat,exist_ok=True)
                    session['namecat'] = session['user_folder']+"/ontology/"+str(timeextension)+"_0_"+namecat+"/"+namecat
                    if namecat=='addictionnew':
                        with open("addiction.onto","r") as f1:
                            with open(session['namecat']+".onto", "w") as f2:
                                for line in f1:
                                    f2.write(line)  
                    else: 
                        f= open(session['namecat']+".onto","w")
                        dict_onto={}
                else:
                    session['namecat'] = session['user_folder']+"/ontology/"+namecat_filename+'/'+namecat

                onto_cont=open(session['namecat']+".onto",'r').read()
                if onto_cont=='':
                    dict_onto={}
                else:
                    dict_onto=ast.literal_eval(onto_cont)

                flag_kw=0
                if (',' in maincat) or (',' in subcat):
                    flash("Only one word can be added to the category and subcategory at a time.","inval")
                elif maincat in dict_onto.keys():  # Layer 2, main category 
                    if subcat in dict_onto[maincat].keys():  # Layer 3, keywords shown in results 
                        keycat_ls = keycat.split('|')
                        for kw in str.split(next(iter(dict_onto[maincat][subcat])), '|'):  # Layer 4, synonyms
                            for keycat_word in keycat_ls:
                                if kw==keycat_word:
                                    flash("\""+kw+"\" is already in keywords under the subcategory \""+ subcat \
                                        + "\" that is under the category \""+ maincat+"\"","inval")
                                    flag_kw=1
                        if flag_kw==0:
                            dict_onto[maincat][subcat]= '{'+next(iter(dict_onto[maincat][subcat]))+'|'+keycat+'}'
                            dict_onto=str(dict_onto).replace('\'{','{\'')
                            dict_onto=str(dict_onto).replace('}\'','\'}')
                            dict_onto=str(dict_onto).replace('}},','}},\n')
                            with open(session['namecat']+'.onto', 'w') as file3:
                                file3.write(str(dict_onto))
                    else:
                        dict_onto[maincat][subcat]='{'+subcat+'|'+keycat+'}'
                        dict_onto=str(dict_onto).replace('\'{','{\'')
                        dict_onto=str(dict_onto).replace('}\'','\'}')
                        dict_onto=str(dict_onto).replace('}},','}},\n')
                        with open(session['namecat']+'.onto', 'w') as file3:
                            file3.write(str(dict_onto))
                else:
                    dict_onto[maincat]= '{'+subcat+'\': {\''+keycat+'}'+'}'
                    dict_onto=str(dict_onto).replace('\"{','{\'')
                    dict_onto=str(dict_onto).replace('}\"','\'}')
                    dict_onto=str(dict_onto).replace('\'{','{\'')
                    dict_onto=str(dict_onto).replace('}\'','\'}')
                    dict_onto=str(dict_onto).replace('}},','}},\n')
                    with open(session['namecat']+'.onto', 'w') as file3:
                        file3.write(str(dict_onto))
            else:
                if namecat=='addiction':
                    flash("You must login to change the addiction ontology.")
                else:
                    flash("You must login to create a new ontology.")
        
        if request.form['submit'] == 'remove':
            if ('email' in session):
                session['namecat']=namecat
                if (namecat=='addiction'):
                    flash("You cannot change addiction keywords but a new ontology will be saved as 'addictionnew', instead","inval")
                    namecat='addictionnew'
                    session['namecat']=namecat
                session['user_folder'] = datadir+"/user/"+str(session['hashed_email'])
                dirlist = os.listdir(session['user_folder']+"/ontology/")
                for filename in dirlist:
                    onto_name = filename.split('_0_')[1]
                    if onto_name==namecat:
                        namecat_exist=1
                        namecat_filename=filename
                        break
                if namecat_exist==0:
                    os.makedirs(session['user_folder']+"/ontology/"+str(timeextension)+"_0_"+namecat,exist_ok=True)
                    session['namecat'] = session['user_folder']+"/ontology/"+str(timeextension)+"_0_"+namecat+"/"+namecat
                    if namecat=='addictionnew':
                        with open("addiction.onto","r") as f1:
                            with open(session['namecat']+".onto", "w") as f2:
                                for line in f1:
                                    f2.write(line)  
                    else: 
                        f= open(session['namecat']+".onto","w")
                        dict_onto={}

                else:
                    session['namecat'] = session['user_folder']+"/ontology/"+namecat_filename+'/'+namecat

                onto_cont=open(session['namecat']+".onto",'r').read()
                if onto_cont=='':
                    dict_onto={}
                else:
                    dict_onto=ast.literal_eval(onto_cont)
                
                flag_kw=0
                if maincat in dict_onto.keys():  # Layer 2, main category 
                    if subcat in dict_onto[maincat].keys():  # Layer 3, keywords shown in results 
                        for kw in str.split(next(iter(dict_onto[maincat][subcat])), '|'):
                            keycat_ls = keycat.split('|')
                            for keycat_word in keycat_ls:  # Layer 4, synonyms
                                if kw==keycat_word:
                                    dict_onto[maincat][subcat]=re.sub(r'\|'+keycat_word+'\'', '\'', str(dict_onto[maincat][subcat]))
                                    dict_onto[maincat][subcat]=re.sub(r'\''+keycat_word+'\|', '\'', str(dict_onto[maincat][subcat]))
                                    dict_onto[maincat][subcat]=re.sub(r'\|'+keycat_word+'\|', '|', str(dict_onto[maincat][subcat]))
                                    dict_onto[maincat][subcat]=re.sub(r'\''+keycat_word+'\'', '', str(dict_onto[maincat][subcat]))
                                    flag_kw=1
                        if '{}' in dict_onto[maincat][subcat]:
                            dict_onto[maincat]=re.sub(r', \''+subcat+'\': \'{}\' ', '', str(dict_onto[maincat]))
                            dict_onto[maincat]=re.sub(r'\''+subcat+'\': \'{}\', ', '', str(dict_onto[maincat]))
                            dict_onto[maincat]=re.sub(r'\''+subcat+'\': \'{}\'', '', str(dict_onto[maincat]))
                        if '{}' in dict_onto[maincat]:
                            dict_onto=re.sub(r', \''+maincat+'\': \'{}\'', '', str(dict_onto))                    
                        dict_onto=str(dict_onto).replace('\"{','{')
                        dict_onto=str(dict_onto).replace('}\"','}')
                        dict_onto=str(dict_onto).replace('\'{','{')
                        dict_onto=str(dict_onto).replace('}\'','}')                   
                        with open(session['namecat']+'.onto', 'w') as file3:
                            file3.write(str(dict_onto))
                        if flag_kw==0:
                            flash("\""+keycat+"\" is not a keyword.","inval")
                    else:
                        flash("\""+subcat+"\" is not a subcategory.","inval")
                else:
                    flash("\""+subcat+"\" is not a category.","inval")  
            else:
                if namecat=='addiction':
                    flash("You must login to change the addiction ontology.")
                else:
                    flash("You must login to create a new ontology.")         

    if 'namecat' in session:
        file2 = open(session['namecat']+".onto","r")
        onto_cont=file2.read()
        if onto_cont=='':
            dict_onto={}
        else:
            dict_onto=ast.literal_eval(onto_cont)
    else:
        session['namecat']='addiction'
        file2 = open(session['namecat']+".onto","r")
        onto_cont=file2.read()
        dict_onto=ast.literal_eval(onto_cont)
    name_to_html = str(session['namecat']).split('/')[-1]

    if ('email' in session):
        ontoarchive()
        onto_len_dir = session['onto_len_dir']
        onto_list = session['onto_list']
    else:
        onto_len_dir=0
        onto_list=''
    return render_template('ontology.html',dict_onto=dict_onto, namecat=name_to_html, onto_len_dir=onto_len_dir, onto_list=onto_list)


@app.route("/ontoarchive")
def ontoarchive():
    session['onto_len_dir'] = 0
    session['onto_list'] = ''
    if ('email' in session):
        if os.path.exists(datadir+"/user/"+str(session['hashed_email'])+"/ontology") == False:
            flash("Ontology history doesn't exist!")
            return render_template('index.html',onto_len_dir=session['onto_len_dir'], onto_list=session['onto_list'])
        else:
            session['user_folder'] = datadir+"/user/"+str(session['hashed_email'])
    else:
        flash("You logged out!")
        onto_len_dir = 0
        onto_list = ''
        onto_cont=open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
        return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)

    session_id=session['id']
    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    dirlist = sorted_alphanumeric(os.listdir(session['user_folder']+"/ontology/"))  
    onto_folder_list = []
    onto_directory_list = []
    onto_list=[]

    for filename in dirlist:
        onto_folder_list.append(filename)
        onto_name = filename.split('_0_')[1]
        onto_name = onto_name.replace('_', ', ')
        onto_list.append(onto_name)
        onto_name=""
        filename=filename[0:4]+"-"+filename[5:7]+"-"+filename[8:13]+":"+filename[14:16]+":"+filename[17:19]
        onto_directory_list.append(filename)

    onto_len_dir = len(onto_directory_list)
    session['onto_len_dir'] = onto_len_dir
    session['onto_list'] = onto_list
    message3="<ul><li> Click on the Date/Time to view archived results. <li>The Date/Time are based on US Central time zone.</ul> "
    return render_template('ontoarchive.html', onto_len_dir=onto_len_dir, onto_list = onto_list, onto_folder_list=onto_folder_list, onto_directory_list=onto_directory_list, session_id=session_id, message3=message3)


# Remove an ontology folder
@app.route('/removeonto', methods=['GET', 'POST'])
def removeonto():
    if('email' in session):
        remove_folder = request.args.get('remove_folder')
        shutil.rmtree(datadir+"/user/"+str(session['hashed_email']+"/ontology/"+remove_folder), ignore_errors=True)
        return redirect(url_for('ontoarchive'))
    else:
        flash("You logged out!")
        onto_len_dir = 0
        onto_list = ''
        onto_cont=open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
        return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)


@app.route('/progress')
def progress():
    genes=request.args.get('query')
    genes=genes.replace(",", " ")
    genes=genes.replace(";", " ")
    genes=re.sub(r'\bLOC\d*?\b', "", genes, flags=re.I)
    genes=genes.replace(" \'", " \"")
    genes=genes.replace("\' ", "\" ")
    genes=genes.replace("\'", "-")
    genes1 = [f[1:-1] for f in re.findall('".+?"', genes)]
    genes2 = [p for p in re.findall(r'([^""]+)',genes) if p not in genes1]
    genes2_str = ''.join(genes2)
    genes2 = genes2_str.split()
    genes3 = genes1 + genes2
    genes = [re.sub("\s+", '-', s) for s in genes3]

    # Only 1-200 terms are allowed
    if len(genes)>=200:
        if ('email' in session):
            onto_len_dir = session['onto_len_dir']
            onto_list = session['onto_list']
        else: 
            onto_len_dir = 0
            onto_list = ''
        onto_cont=open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
        message="<span class='text-danger'>Up to 200 terms can be searched at a time</span>"
        return render_template('index.html' ,onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto, message=message)
    
    if len(genes)==0:
        if ('email' in session):
            onto_len_dir = session['onto_len_dir']
            onto_list = session['onto_list']
        else: 
            onto_len_dir = 0
            onto_list = ''
        onto_cont=open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
        message="<span class='text-danger'>Please enter a search term </span>"
        return render_template('index.html',onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto, message=message)
    
    tf_path=tempfile.gettempdir()
    genes_for_folder_name =""
    if len(genes) == 1:
        marker = ""
        genes_for_folder_name =str(genes[0])
    elif len(genes) == 2:
        marker = ""
        genes_for_folder_name =str(genes[0])+"_"+str(genes[1])
    elif len(genes) == 3:
        marker = ""
        genes_for_folder_name =str(genes[0])+"_"+str(genes[1])+"_"+str(genes[2])
    else:
        genes_for_folder_name =str(genes[0])+"_"+str(genes[1])+"_"+str(genes[2])
        marker="_m"

    # Generate a unique session ID depending on timestamp to track the results 
    timestamp = datetime.utcnow().replace(microsecond=0)
    timestamp = timestamp.replace(tzinfo=pytz.utc)
    timestamp = timestamp.astimezone(pytz.timezone("America/Chicago"))
    timeextension = str(timestamp)
    timeextension = timeextension.replace(':', '_')
    timeextension = timeextension.replace('-', '_')
    timeextension = timeextension.replace(' ', '_')
    timeextension = timeextension.replace('_06_00', '')
    session['timeextension'] = timeextension
    namecat_exist=0

    # Create a folder for the search
    if ('email' in session):
        try:
            namecat=session['namecat']
        except:
            namecat = 'addiction'
            session['namecat'] = namecat
        if namecat=='choose your ontology' or namecat=='addiction' or namecat == 'addiction':
            session['namecat']='addiction'
            onto_cont=open("addiction.onto","r").read()
            dictionary=ast.literal_eval(onto_cont)
            search_type = request.args.getlist('type')
            if (search_type == []):
                search_type = ['GWAS', 'function', 'addiction', 'drug', 'brain', 'stress', 'psychiatric', 'cell']
            session['search_type'] = search_type
        else:
            dirlist = os.listdir(session['user_folder']+"/ontology/")
            for filename in dirlist:
                onto_name = filename.split('_0_')[1]
                if onto_name==namecat:
                    namecat_exist=1
                    namecat_filename=filename
                    break
            if (namecat_exist==1):
                session['namecat'] = session['user_folder']+"/ontology/"+namecat_filename+'/'+namecat
                onto_cont=open(session['namecat']+".onto","r").read()
                dict_onto=ast.literal_eval(onto_cont)
                search_type = request.args.getlist('type')
                if (search_type == []):
                    search_type = list(dict_onto.keys())
                session['search_type'] = search_type

        # Save the ontology name in the user search history table
        if session['namecat']=='addiction':
            onto_name_archive = session['namecat']
        elif ('/' in session['namecat']):
            onto_name_archive = session['namecat'].split('/')[-1]
        else:
            onto_name_archive = '-'

        os.makedirs(datadir + "/user/"+str(session['hashed_email'])+"/"+str(timeextension)+"_0_"+genes_for_folder_name+marker+"_0_"+onto_name_archive,exist_ok=True)
        session['path_user'] = datadir+"/user/"+str(session['hashed_email'])+"/"+str(timeextension)+"_0_"+genes_for_folder_name+marker+"_0_"+onto_name_archive+"/"
        session['rnd'] = timeextension+"_0_"+genes_for_folder_name+marker+"_0_"+onto_name_archive
        rnd = session['rnd']
    else:
        rnd = "tmp" + ''.join(random.choice(string.ascii_letters) for x in range(6)) 
        session['path']=tf_path+ "/" + rnd
        os.makedirs(session['path'])
        search_type = request.args.getlist('type')

        if (search_type == []):
            search_type = ['GWAS', 'function', 'addiction', 'drug', 'brain', 'stress', 'psychiatric', 'cell']
        session['search_type'] = search_type
    genes_session = ''

    for gen in genes:
        genes_session += str(gen) + "_"
    genes_session = genes_session[:-1]
    session['query']=genes
    return render_template('progress.html', url_in="search", url_out="cytoscape/?rnd="+rnd+"&genequery="+genes_session)


@app.route("/search")
def search():
    genes=session['query']
    percent_ratio=len(genes)+1

    if(len(genes)==1):
        percent_ratio=2
    timeextension=session['timeextension']
    percent=100/percent_ratio-0.00000001 # 7 categories + 1 at the beginning

    if ('email' in session):
        sessionpath = session['path_user'] + timeextension
        path_user=session['path_user']
    else:
        sessionpath = session['path']
        path_user=session['path']+"/"

    snt_file=sessionpath+"_snt"
    cysdata=open(sessionpath+"_cy","w+")
    sntdata=open(snt_file,"w+")
    zeroLinkNode=open(sessionpath+"_0link","w+")
    search_type = session['search_type']
    temp_nodes = ""
    json_nodes = "{\"data\":["
    
    n_num=0
    d={}
    nodecolor={}
    nodecolor['GWAS'] = "hsl(0, 0%, 70%)"
    nodes_list = []

    if 'namecat' in session:
        namecat_flag=1
        ses_namecat = session['namecat']
        onto_cont = open(session['namecat']+".onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)

        for ky in dict_onto.keys():
            nodecolor[ky] = "hsl("+str((n_num+1)*int(360/len(dict_onto.keys())))+", 70%, 80%)"
            d["nj{0}".format(n_num)]=generate_nodes_json(dict_onto[ky],str(ky),nodecolor[ky])
            n_num+=1

            if (ky in search_type):
                temp_nodes += generate_nodes(dict_onto[ky],str(ky),nodecolor[ky])

                for nd in dict_onto[ky]:
                    nodes_list.append(nd)
                json_nodes += generate_nodes_json(dict_onto[ky],str(ky),nodecolor[ky] )
        d["nj{0}".format(n_num)]=''
    else:
        namecat_flag=0
        for ky in dictionary.keys():
            nodecolor[ky] = "hsl("+str((n_num+1)*int(360/len(dictionary.keys())))+", 70%, 80%)"
            d["nj{0}".format(n_num)]=generate_nodes_json(dictionary[ky],str(ky),nodecolor[ky])
            n_num+=1

            if (ky in search_type):
                temp_nodes += generate_nodes(dictionary[ky],str(ky),nodecolor[ky])

                for nd in dictionary[ky]:
                    nodes_list.append(nd)
                json_nodes += generate_nodes_json(dictionary[ky],str(ky),nodecolor[ky])
        d["nj{0}".format(n_num)]=''
    
    json_nodes = json_nodes[:-2]
    json_nodes =json_nodes+"]}"
    def generate(genes, tf_name):
        with app.test_request_context():
            sentences=str()
            edges=str()
            nodes = temp_nodes
            progress=0
            searchCnt=0
            nodesToHide=str()
            json_edges = str()           
            #genes_or = ' [tiab] or '.join(genes)
            all_d=''

            if namecat_flag==1:
                onto_cont = open(ses_namecat+".onto","r").read()
                dict_onto=ast.literal_eval(onto_cont)

                for ky in dict_onto.keys():
                    if (ky in search_type):
                        all_d_ls=undic(list(dict_onto[ky].values()))
                        all_d = all_d+'|'+all_d_ls
            else:
                for ky in dictionary.keys():
                    if (ky in search_type):
                        all_d_ls=undic(list(dictionary[ky].values()))
                        all_d = all_d+'|'+all_d_ls
            all_d=all_d[1:]
            if ("GWAS" in search_type):
                datf = pd.read_csv('./utility/gwas_used.csv',sep='\t')
            progress+=percent
            yield "data:"+str(progress)+"\n\n"
            for gene in genes:
                abstracts_raw = getabstracts(gene,all_d)
                #print(abstracts_raw)
                sentences_ls=[]

                for row in abstracts_raw.split("\n"):
                    tiab=row.split("\t")
                    pmid = tiab.pop(0)
                    tiab= " ".join(tiab)
                    sentences_tok = sent_tokenize(tiab)
                    for sent_tok in sentences_tok:
                        sent_tok = pmid + ' ' + sent_tok
                        sentences_ls.append(sent_tok)
                gene=gene.replace("-"," ")
                
                geneEdges = ""

                if namecat_flag==1:
                    onto_cont = open(ses_namecat+".onto","r").read()
                    dict_onto=ast.literal_eval(onto_cont)
                else:
                    dict_onto = dictionary

                for ky in dict_onto.keys():
                    if (ky in search_type):
                        if (ky=='addiction') and ('addiction' in dict_onto.keys())\
                            and ('drug' in dict_onto.keys()) and ('addiction' in dict_onto['addiction'].keys())\
                            and ('aversion' in dict_onto['addiction'].keys()) and ('intoxication' in dict_onto['addiction'].keys()):
                            #addiction terms must present with at least one drug
                            addiction_flag=1
                            #addiction=undic0(addiction_d) +") AND ("+undic0(drug_d)
                            sent=gene_category(gene, addiction_d, "addiction", sentences_ls,addiction_flag,dict_onto)
                            if ('addiction' in search_type):
                                geneEdges += generate_edges(sent, tf_name)
                                json_edges += generate_edges_json(sent, tf_name)
                        else:
                            addiction_flag=0
                            if namecat_flag==1:
                                onto_cont = open(ses_namecat+".onto","r").read()
                                dict_onto=ast.literal_eval(onto_cont)
                                #ky_d=undic(list(dict_onto[ky].values()))    
                                sent=gene_category(gene,ky,str(ky), sentences_ls, addiction_flag,dict_onto)
                            else:
                                #ky_d=undic(list(dict_onto[ky].values()))
                                #print(sentences_ls)
                                sent=gene_category(gene,ky,str(ky), sentences_ls, addiction_flag,dict_onto)
                                #print(sent)
                            yield "data:"+str(progress)+"\n\n"
                            
                            geneEdges += generate_edges(sent, tf_name)
                            json_edges += generate_edges_json(sent, tf_name)                
                        sentences+=sent
                if ("GWAS" in search_type):
                    gwas_sent=[]
                    datf_sub1 = datf[datf['REPORTED GENE(S)'].str.contains('(?:\s|^)'+gene+'(?:\s|$)', flags=re.IGNORECASE)
                                    | (datf['MAPPED_GENE'].str.contains('(?:\s|^)'+gene+'(?:\s|$)', flags=re.IGNORECASE))]
                    for nd2 in dict_onto['GWAS'].keys():
                        for nd1 in dict_onto['GWAS'][nd2]:    
                            for nd in nd1.split('|'):
                                gwas_text=''
                                datf_sub = datf_sub1[datf_sub1['DISEASE/TRAIT'].str.contains('(?:\s|^)'+nd+'(?:\s|$)', flags=re.IGNORECASE)]
                                    #& (datf['REPORTED GENE(S)'].str.contains('(?:\s|^)'+gene+'(?:\s|$)', flags=re.IGNORECASE)
                                    #| (datf['MAPPED_GENE'].str.contains('(?:\s|^)'+gene+'(?:\s|$)', flags=re.IGNORECASE)))]
                                if not datf_sub.empty:
                                    for index, row in datf_sub.iterrows():
                                        gwas_text = "SNP:<b>"+str(row['SNPS'])+"</b>, P value: <b>"+str(row['P-VALUE'])\
                                            +"</b>, Disease/trait:<b> "+str(row['DISEASE/TRAIT'])+"</b>, Mapped trait:<b> "\
                                            +str(row['MAPPED_TRAIT'])+"</b><br>"
                                        gwas_sent.append(gene+"\t"+"GWAS"+"\t"+nd+"_GWAS\t"+str(row['PUBMEDID'])+"\t"+gwas_text)
                    cys, gwas_json, sn_file = searchArchived('GWAS', gene , 'json',gwas_sent, path_user)
                    with open(path_user+"gwas_results.tab", "a") as gwas_edges:
                        gwas_edges.write(sn_file)
                    geneEdges += cys
                    json_edges += gwas_json  
                # report progress immediately
                progress+=percent
                yield "data:"+str(progress)+"\n\n"
                                    
                if len(geneEdges) >0:
                    edges+=geneEdges
                    nodes+="{ data: { id: '" + gene +  "', nodecolor:'#E74C3C', fontweight:700, url:'/synonyms?node="+gene+"'} },\n"
                else:
                    nodesToHide+=gene +  " "

                searchCnt+=1
                if (searchCnt==len(genes)):
                    progress=100
                    sntdata.write(sentences)
                    sntdata.close()
                    cysdata.write(nodes+edges)               
                    cysdata.close()
                    zeroLinkNode.write(nodesToHide)
                    zeroLinkNode.close()
                yield "data:"+str(progress)+"\n\n"

           # Edges in json format
            json_edges="{\"data\":["+json_edges
            json_edges = json_edges[:-2]
            json_edges =json_edges+"]}"

            # Write edges to txt file in json format also in user folder
            with open(path_user+"edges.json", "w") as temp_file_edges:
                temp_file_edges.write(json_edges) 
    with open(path_user+"nodes.json", "w") as temp_file_nodes:
        temp_file_nodes.write(json_nodes)
    return Response(generate(genes, snt_file), mimetype='text/event-stream')


@app.route("/tableview/")
def tableview():
    genes_url=request.args.get('genequery')
    rnd_url=request.args.get('rnd')
    tf_path=tempfile.gettempdir()

    if ('email' in session):
        filename = rnd_url.split("_0_")[0]
        genes_session_tmp = datadir+"/user/"+str(session['hashed_email'])+"/"+rnd_url+"/"+filename
        gene_url_tmp = "/user/"+str(session['hashed_email'])+"/"+rnd_url

        try:
            with open(datadir+gene_url_tmp+"/nodes.json") as jsonfile:
                jnodes = json.load(jsonfile)
        except FileNotFoundError:
            flash("You logged out!")
            onto_len_dir = 0
            onto_list = ''
            onto_cont=open("addiction.onto","r").read()
            dict_onto=ast.literal_eval(onto_cont)
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)

        jedges =''
        file_edges = open(datadir+gene_url_tmp +'/edges.json', 'r')
        for line in file_edges.readlines():
            if ':' not in line:
                nodata_temp = 1
            else: 
                nodata_temp = 0
                with open(datadir+gene_url_tmp +"/edges.json") as edgesjsonfile:
                    jedges = json.load(edgesjsonfile)
                break
    else:
        genes_session_tmp=tf_path+"/"+rnd_url
        gene_url_tmp = genes_session_tmp
        try:
            with open(gene_url_tmp+"/nodes.json") as jsonfile:
                jnodes = json.load(jsonfile)
        except FileNotFoundError:
            flash("You logged out!")
            onto_len_dir = 0
            onto_list = ''
            onto_cont=open("addiction.onto","r").read()
            dict_onto=ast.literal_eval(onto_cont)
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)
        jedges =''
        file_edges = open(gene_url_tmp +'/edges.json', 'r')
        for line in file_edges.readlines():
            if ':' not in line:
                nodata_temp = 1
            else: 
                nodata_temp = 0
                with open(gene_url_tmp +"/edges.json") as edgesjsonfile:
                    jedges = json.load(edgesjsonfile)
                break
    genename=genes_url.split("_")
    if len(genename)>3:
        genename = genename[0:3]
        added = ",..."
    else:
        added = ""
    gene_name = str(genename)[1:]
    gene_name=gene_name[:-1]
    gene_name=gene_name.replace("'","")
    gene_name = gene_name+added
    num_gene = gene_name.count(',')+1

    message3="<ul><li> <font color=\"#E74C3C\">Click on the abstract count to read sentences linking the keyword and the gene</font>  <li> Click on a keyword to see the terms included in the search. <li>View the results in <a href='\\cytoscape/?rnd={}&genequery={}'\ ><b> a graph.</b></a> </ul> Links will be preserved when the table is copy-n-pasted into a spreadsheet.".format(rnd_url,genes_url)
    return render_template('tableview.html', genes_session_tmp = genes_session_tmp, nodata_temp=nodata_temp, num_gene=num_gene, jedges=jedges, jnodes=jnodes,gene_name=gene_name, message3=message3, rnd_url=rnd_url, genes_url=genes_url)


# Table for the zero abstract counts
@app.route("/tableview0/")
def tableview0():
    genes_url=request.args.get('genequery')
    rnd_url=request.args.get('rnd')
    tf_path=tempfile.gettempdir()

    if ('email' in session):
        filename = rnd_url.split("_0_")[0]
        genes_session_tmp = datadir+"/user/"+str(session['hashed_email'])+"/"+rnd_url+"/"+filename
        gene_url_tmp = "/user/"+str(session['hashed_email'])+"/"+rnd_url
        try:
            with open(datadir+gene_url_tmp+"/nodes.json") as jsonfile:
                jnodes = json.load(jsonfile)
        except FileNotFoundError:
            flash("You logged out!")
            onto_len_dir = 0
            onto_list = ''
            onto_cont=open("addiction.onto","r").read()
            dict_onto=ast.literal_eval(onto_cont)
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)

        jedges =''
        file_edges = open(datadir+gene_url_tmp+'/edges.json', 'r')
        for line in file_edges.readlines():
            if ':' not in line:
                nodata_temp = 1
            else: 
                nodata_temp = 0
                with open(datadir+gene_url_tmp+"/edges.json") as edgesjsonfile:
                    jedges = json.load(edgesjsonfile)
                break
    else:
        genes_session_tmp=tf_path+"/"+rnd_url
        gene_url_tmp = genes_session_tmp
        try:
            with open(gene_url_tmp+"/nodes.json") as jsonfile:
                jnodes = json.load(jsonfile)
        except FileNotFoundError:
            flash("You logged out!")
            onto_len_dir = 0
            onto_list = ''
            onto_cont=open("addiction.onto","r").read()
            dict_onto=ast.literal_eval(onto_cont)
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)

        jedges =''
        file_edges = open(gene_url_tmp+'/edges.json', 'r')
        for line in file_edges.readlines():
            if ':' not in line:
                nodata_temp = 1
            else: 
                nodata_temp = 0
                with open(gene_url_tmp+"/edges.json") as edgesjsonfile:
                    jedges = json.load(edgesjsonfile)
                break
    genes_url=request.args.get('genequery')
    genename=genes_url.split("_")
    if len(genename)>3:
        genename = genename[0:3]
        added = ",..."
    else:
        added = ""

    gene_name = str(genename)[1:]
    gene_name=gene_name[:-1]
    gene_name=gene_name.replace("'","")
    gene_name = gene_name+added
    num_gene = gene_name.count(',')+1
    message4="<b> Notes: </b><li> These are the keywords that have <b>zero</b> abstract counts. <li>View all the results in <a href='\\cytoscape/?rnd={}&genequery={}'><b> a graph.</b></a> ".format(rnd_url,genes_url)
    return render_template('tableview0.html',nodata_temp=nodata_temp, num_gene=num_gene, jedges=jedges, jnodes=jnodes,gene_name=gene_name, message4=message4)


@app.route("/userarchive")
def userarchive():
    onto_len_dir = 0
    onto_list = ''
    onto_cont=open("addiction.onto","r").read()
    dict_onto=ast.literal_eval(onto_cont)

    if ('email' in session):
        if os.path.exists(datadir+"/user/"+str(session['hashed_email'])) == False:
            flash("Search history doesn't exist!")
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)
        else:
            session['user_folder'] = datadir+"/user/"+str(session['hashed_email'])
    else:
        onto_name_archive=''
        flash("You logged out!")
        onto_len_dir = 0
        onto_list = ''
        onto_cont=open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
        return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)

    session_id=session['id']
    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)
    dirlist = sorted_alphanumeric(os.listdir(session['user_folder']))  
    folder_list = []
    directory_list = []
    gene_list=[]
    onto_list=[]

    for filename in dirlist:
        if ('_0_'  in filename):
            folder_list.append(filename)
            gene_name = filename.split('_0_')[1]
            onto_name = filename.split('_0_')[2]
            if gene_name[-2:] == '_m':
                gene_name = gene_name[:-2]
                gene_name = gene_name + ", ..."
            gene_name = gene_name.replace('_', ', ')
            gene_list.append(gene_name)
            onto_list.append(onto_name)
            onto_name=""
            gene_name=""
            filename=filename[0:4]+"-"+filename[5:7]+"-"+filename[8:13]+":"+filename[14:16]+":"+filename[17:19]
            directory_list.append(filename)
    len_dir = len(directory_list)
    message3="<ul><li> Click on the Date/Time to view archived results. <li>The Date/Time are based on US Central time zone.</ul> "
    return render_template('userarchive.html', len_dir=len_dir, gene_list = gene_list, onto_list = onto_list, folder_list=folder_list, directory_list=directory_list, session_id=session_id, message3=message3)


# Remove the search directory
@app.route('/remove', methods=['GET', 'POST'])
def remove():
    if('email' in session):
        remove_folder = request.args.get('remove_folder')
        shutil.rmtree(datadir+"/user/"+str(session['hashed_email']+"/"+remove_folder), ignore_errors=True)
        return redirect(url_for('userarchive'))
    else:
        flash("You logged out!")
        onto_len_dir = 0
        onto_list = ''
        onto_cont=open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
        return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)


@app.route('/date', methods=['GET', 'POST'])
def date():
    select_date = request.args.get('selected_date')
    # Open the cache folder for the user
    tf_path=datadir+"/user"
    if ('email' in session):
        time_extension = str(select_date)
        time_extension = time_extension.split('_0_')[0]
        gene_name1 = str(select_date).split('_0_')[1]
        time_extension = time_extension.replace(':', '_')
        time_extension = time_extension.replace('-', '_')
        session['user_folder'] = tf_path+"/"+str(session['hashed_email'])
        genes_session_tmp = tf_path+"/"+str(session['hashed_email'])+"/"+select_date+"/"+time_extension
        with open(tf_path+"/"+str(session['hashed_email'])+"/"+select_date+"/nodes.json", "r") as jsonfile:
            jnodes = json.load(jsonfile)
        jedges =''
        file_edges = open(tf_path+"/"+str(session['hashed_email'])+"/"+select_date+"/edges.json", "r")
        for line in file_edges.readlines():
            if ':' not in line:
                nodata_temp = 1
            else:
                nodata_temp = 0
                with open(tf_path+"/"+str(session['hashed_email'])+"/"+select_date+"/edges.json", "r") as edgesjsonfile:
                    jedges = json.load(edgesjsonfile)
                break
        gene_list_all=[]
        gene_list=[]
        if nodata_temp == 0:
            for p in jedges['data']:
                if p['source'] not in gene_list:
                    gene_list_all.append(p['source'])
                    gene_list.append(p['source'])
            if len(gene_list)>3:
                gene_list = gene_list[0:3]
                added = ",..."
            else:
                added = ""
            gene_name = str(gene_list)[1:]
            gene_name=gene_name[:-1]
            gene_name=gene_name.replace("'","")
            gene_name = gene_name+added
            num_gene = gene_name.count(',')+1
        else:
            gene_name1 = gene_name1.replace("_", ", ")
            gene_name = gene_name1
            num_gene = gene_name1.count(',')+1
            for i in range(0,num_gene):
                gene_list.append(gene_name1.split(',')[i])
        genes_session = ''
        for gen in gene_list_all:
            genes_session += str(gen) + "_"
        genes_session = genes_session[:-1]
    else:
        flash("You logged out!")
        onto_len_dir = 0
        onto_list = ''
        onto_cont=open("addiction.onto","r").read()
        dict_onto=ast.literal_eval(onto_cont)
        return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)
    message3="<ul><li> <font color=\"#E74C3C\">Click on the abstract count to read sentences linking the keyword and the gene</font> <li> Click on a keyword to see the terms included in the search. <li>View the results in <a href='\\cytoscape/?rnd={}&genequery={}'\ ><b> a graph.</b></a> </ul> Links will be preserved when the table is copy-n-pasted into a spreadsheet.".format(select_date,genes_session)
    return render_template('tableview.html',nodata_temp=nodata_temp, num_gene=num_gene,genes_session_tmp = genes_session_tmp, rnd_url=select_date ,jedges=jedges, jnodes=jnodes,gene_name=gene_name, genes_url=genes_session, message3=message3)

@app.route('/cytoscape/')
def cytoscape():
    genes_url=request.args.get('genequery')
    rnd_url=request.args.get('rnd')
    tf_path=tempfile.gettempdir()
    genes_session_tmp=tf_path + "/" + genes_url
    rnd_url_tmp=tf_path +"/" + rnd_url
    message2="<ul><li><font color=\"#E74C3C\">Click on a line to read the sentences </font> <li>Click on a keyword to see the terms included in the search<li>Hover a pointer over a node to hide other links <li>Move the nodes around to adjust visibility <li> Reload the page to restore the default layout<li>View the results in <a href='\\tableview/?rnd={}&genequery={}'\ ><b>a table. </b></a></ul>".format(rnd_url,genes_url)
    
    if ('email' in session):
        filename = rnd_url.split("_0_")[0]
        rnd_url_tmp = datadir+"/user/"+str(session['hashed_email'])+"/"+rnd_url+"/"+filename
        try:
            with open(rnd_url_tmp+"_cy","r") as f:
                elements=f.read()
        except FileNotFoundError:
            flash("You logged out!")
            onto_len_dir = 0
            onto_list = ''
            onto_cont=open("addiction.onto","r").read()
            dict_onto=ast.literal_eval(onto_cont)
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)

        with open(rnd_url_tmp+"_0link","r") as z:
            zeroLink=z.read()
            if (len(zeroLink)>0):
                message2+="<span style=\"color:darkred;\">No result was found for these genes: " + zeroLink + "</span>"
    else:
        rnd_url_tmp=tf_path +"/" + rnd_url
        try:
            rnd_url_tmp.replace("\"", "")
            with open(rnd_url_tmp+"_cy","r") as f:
                elements=f.read()
        except FileNotFoundError:
            flash("You logged out!")
            onto_len_dir = 0
            onto_list = ''
            onto_cont=open("addiction.onto","r").read()
            dict_onto=ast.literal_eval(onto_cont)
            return render_template('index.html', onto_len_dir=onto_len_dir, onto_list=onto_list, ontol = 'addiction', dict_onto = dict_onto)

        with open(rnd_url_tmp+"_0link","r") as z:
            zeroLink=z.read()
            if (len(zeroLink)>0):
                message2+="<span style=\"color:darkred;\">No result was found for these genes: " + zeroLink + "</span>"
    return render_template('cytoscape.html', elements=elements, message2=message2)


@app.route("/sentences")
def sentences():
    def predict_sent(sent_for_pred):
        max_length = 64
        tokens = clean_doc(sent_for_pred, vocab)
        tokens = [w for w in tokens if w in vocab]
        # convert to line
        line = ' '.join(tokens)
        line = [line]
        tokenized_sent = tokenizer.texts_to_sequences(line)
        tokenized_sent = pad_sequences(tokenized_sent, maxlen=max_length, padding='post') 
        predict_sent = model.predict(tokenized_sent, verbose=0)
        percent_sent = predict_sent[0,0]
        if round(percent_sent) == 0:
            return 'neg'
        else:
            return 'pos'
    pmid_list=[]
    pmid_string=''
    edge=request.args.get('edgeID')
    (tf_name, gene0, cat0)=edge.split("|")

    if(cat0=='stress'):
        model = create_model(23154, 64)
        model.load_weights("./nlp/weights.ckpt")
    out3=""
    out_pos = ""
    out_neg = ""
    num_abstract = 0
    stress_cellular = "<br><br><br>"+"</ol><b>Sentence(s) describing celluar stress (classified using a deep learning model):</b><hr><ol>"
    stress_systemic = "<b></ol>Sentence(s) describing systemic stress (classified using a deep learning model):</b><hr><ol>"
    with open(tf_name, "r") as df:
        all_sents=df.read()

    for sent in all_sents.split("\n"):
        if len(sent.strip())!=0:
            (gene,nouse,cat, pmid, text)=sent.split("\t")
            if (gene.upper() == gene0.upper() and cat.upper() == cat0.upper()) :
                out3+= "<li> "+ text + " <a href=\"https://www.ncbi.nlm.nih.gov/pubmed/?term=" + pmid +"\" target=_new>PMID:"+pmid+"<br></a>"
                num_abstract += 1
                if(pmid+cat0 not in pmid_list):
                    pmid_string = pmid_string + ' ' + pmid
                    pmid_list.append(pmid+cat0)
                if(cat0=='stress'):
                    out4 = predict_sent(text)
                    if(out4 == 'pos'):
                        out_pred_pos = "<li> "+ text + " <a href=\"https://www.ncbi.nlm.nih.gov/pubmed/?term=" + pmid +"\" target=_new>PMID:"+pmid+"<br></a>"                    
                        out_pos += out_pred_pos
                    else:
                        out_pred_neg = "<li>"+ text + " <a href=\"https://www.ncbi.nlm.nih.gov/pubmed/?term=" + pmid +"\" target=_new>PMID:"+pmid+"<br></a>"                    
                        out_neg += out_pred_neg
    out1="<h3>"+gene0 + " and " + cat0  + "</h3>\n"
    if len(pmid_list)>1:
        out2 = str(num_abstract) + ' sentences in ' + " <a href=\"https://www.ncbi.nlm.nih.gov/pubmed/?term=" + pmid_string +"\" target=_new>"+ str(len(pmid_list)) + ' studies' +"<br></a>" + "<br><br>"
    else:
        out2 = str(num_abstract) + ' sentence(s) in '+ " <a href=\"https://www.ncbi.nlm.nih.gov/pubmed/?term=" + pmid_string +"\" target=_new>"+ str(len(pmid_list)) + ' study' +"<br></a>" "<br><br>"
    if(out_neg == "" and out_pos == ""):
        out= out1+ out2 +out3
    elif(out_pos != "" and out_neg!=""):
        out = out1 + out2 + stress_systemic+out_pos + stress_cellular + out_neg
    elif(out_pos != "" and out_neg ==""):
        out= out1+ out2 + stress_systemic + out_pos
    elif(out_neg != "" and out_pos == ""):
        out = out1 +out2+stress_cellular+out_neg
    K.clear_session()
    return render_template('sentences.html', sentences="<ol>"+out+"</ol><p>")


# Show the cytoscape graph for one gene from the top gene list
@app.route("/showTopGene")
def showTopGene():
    query=request.args.get('topGene')
    nodesEdges=searchArchived('topGene',query, 'cys','','')[0]
    message2="<li><strong>"+query + "</strong> is one of the top addiction genes. <li> An archived search is shown. Click on the blue circle to update the results and include keywords for brain region and gene function. <strong> The update may take a long time to finish.</strong> "
    return render_template("cytoscape.html", elements=nodesEdges, message="Top addiction genes", message2=message2)


@app.route("/shownode")
def shownode():
    node=request.args.get('node')
    if 'namecat' in session:
        file2 = open(session['namecat']+".onto","r")
        onto_cont=file2.read()
        dict_onto=ast.literal_eval(onto_cont)
        for ky in dict_onto.keys():
            if node in dict_onto[ky].keys():
                out="<p>"+node.upper()+"<hr><li>"+ next(iter(dict_onto[ky][node])).replace("|", "<li>")
    else:
        for ky in dictionary.keys():
            if node in dictionary[ky].keys():
                out="<p>"+node.upper()+"<hr><li>"+ next(iter(dictionary[ky][node])).replace("|", "<li>")
    return render_template('sentences.html', sentences=out+"<p>")


@app.route("/synonyms")
def synonyms():
    node=request.args.get('node')
    node=node.upper()
    allnodes={**genes}
    try:
        synonym_list = list(allnodes[node].split("|")) 
        session['synonym_list'] = synonym_list
        session['main_gene'] = node.upper()
        out="<hr><li>"+ allnodes[node].replace("|", "<li>")
        synonym_list_str = ';'.join([str(syn) for syn in synonym_list]) 
        synonym_list_str +=';' + node
        case = 1
        return render_template('genenames.html', case = case, gene = node.upper(), synonym_list = synonym_list, synonym_list_str=synonym_list_str)
    except:
        try:
            synonym_list = session['synonym_list']
            synonym_list_str = ';'.join([str(syn) for syn in synonym_list]) 
            synonym_list_str +=';' + node
            case = 1
            return render_template('genenames.html', case=case, gene = session['main_gene'] , synonym_list = synonym_list, synonym_list_str=synonym_list_str)
        except:
            case = 2
            return render_template('genenames.html', gene = node, case = case)


@app.route("/startGeneGene")
def startGeneGene():
    session['forTopGene']=request.args.get('forTopGene')
    return render_template('progress.html', url_in="searchGeneGene", url_out="showGeneTopGene")


@app.route("/searchGeneGene")
def gene_gene():
    tmp_ggPMID=session['path']+"_ggPMID"
    gg_file=session['path']+"_ggSent" # Gene_gene
    result_file=session['path']+"_ggResult"

    def generate(query):
        progress=1
        yield "data:"+str(progress)+"\n\n"
        os.system("esearch -db pubmed -query \"" +  query + "\" | efetch -format uid |sort >" + tmp_ggPMID)
        abstracts=os.popen("comm -1 -2 topGene_uniq.pmid " + tmp_ggPMID + " |fetch-pubmed -path "+pubmed_path+ " | xtract -pattern PubmedArticle -element MedlineCitation/PMID,ArticleTitle,AbstractText|sed \"s/-/ /g\"").read()
        os.system("rm "+tmp_ggPMID)
        progress=10
        yield "data:"+str(progress)+"\n\n"
        topGenes=dict()
        out=str()
        hitGenes=dict()
        with open("topGene_symb_alias.txt", "r") as top_f:
            for line in top_f:
                (symb, alias)=line.strip().split("\t")
                topGenes[symb]=alias.replace("; ","|")
        allAbstracts= abstracts.split("\n")
        abstractCnt=len(allAbstracts)
        rowCnt=0

        for row in allAbstracts:
            rowCnt+=1
            if rowCnt/10==int(rowCnt/10):
                progress=10+round(rowCnt/abstractCnt,2)*80
                yield "data:"+str(progress)+"\n\n"
            tiab=row.split("\t")
            pmid = tiab.pop(0)
            tiab= " ".join(tiab)
            sentences = sent_tokenize(tiab)
            ## keep the sentence only if it contains the gene 
            for sent in sentences:
                if findWholeWord(query)(sent):
                    sent=re.sub(r'\b(%s)\b' % query, r'<strong>\1</strong>', sent, flags=re.I)
                    for symb in topGenes:
                        allNames=symb+"|"+topGenes[symb]
                        if findWholeWord(allNames)(sent) :
                            sent=sent.replace("<b>","").replace("</b>","")
                            sent=re.sub(r'\b(%s)\b' % allNames, r'<b>\1</b>', sent, flags=re.I)
                            out+=query+"\t"+"gene\t" + symb+"\t"+pmid+"\t"+sent+"\n"
                            if symb in hitGenes.keys():
                                hitGenes[symb]+=1
                            else:
                                hitGenes[symb]=1
        progress=95
        yield "data:"+str(progress)+"\n\n"
        with open(gg_file, "w+") as gg:
            gg.write(out)
            gg.close()
        results="<h4>"+query+" vs top addiction genes</h4> Click on the number of sentences will show those sentences. Click on the <span style=\"background-color:#FcF3cf\">top addiction genes</span> will show an archived search for that gene.<hr>"
        topGeneHits={}
        for key in hitGenes.keys():
            url=gg_file+"|"+query+"|"+key
            if hitGenes[key]==1:
                sentword="sentence"
            else:
                sentword="sentences"
            topGeneHits[ "<li> <a href=/sentences?edgeID=" + url+ " target=_new>" + "Show " + str(hitGenes[key]) + " " + sentword +" </a> about "+query+" and <a href=/showTopGene?topGene="+key+" target=_gene><span style=\"background-color:#FcF3cf\">"+key+"</span></a>" ]=hitGenes[key]
        topSorted = [(k, topGeneHits[k]) for k in sorted(topGeneHits, key=topGeneHits.get, reverse=True)]
        
        for k,v in topSorted:
            results+=k
        saveResult=open(result_file, "w+")
        saveResult.write(results)
        saveResult.close()
        progress=100
        yield "data:"+str(progress)+"\n\n"
    
    # Start the run
    query=session['forTopGene']
    return Response(generate(query), mimetype='text/event-stream')


@app.route('/showGeneTopGene')
def showGeneTopGene ():
    with open(session['path']+"_ggResult", "r") as result_f:
        results=result_f.read()
    return render_template('sentences.html', sentences=results+"<p><br>")


# Generate a page that lists all the top 150 addiction genes with links to cytoscape graph.
@app.route("/allTopGenes")
def top150genes():
    return render_template("topAddictionGene.html")


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True, port=4200)
