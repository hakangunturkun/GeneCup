#!/bin/env python3 
from nltk.tokenize import sent_tokenize
import os
import re

from nltk.util import pr
from addiction_keywords import *
from gene_synonyms import *
import ast
from flask import session

global pubmed_path

def undic(dic):
    all_s=''
    for s in dic:
        all_s += "|".join(str(e) for e in s)
        all_s +="|"
    all_s=all_s[:-1]
    return all_s

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def getabstracts(gene,query):
    if query[-1] =='s':
        query2 = query+"*"
    else:
        query2 = query+"s*"
    query3 = query2.replace("s|", "s* OR ")
    query4 = query3.replace("|", "s* OR ")
    query="\"(" + query4 + ") AND " + gene + "\""
    abstracts = os.popen("esearch -db pubmed -query " +  query \
        + " | efetch -format uid |fetch-pubmed -path "+ pubmed_path \
        + " | xtract -pattern PubmedArticle -element MedlineCitation/PMID,ArticleTitle,AbstractText|sed \"s/-/ /g\"").read()
    return(abstracts)

sentences_ls=[]
def getSentences(gene, sentences_ls):
    out=str()
    # Keep the sentence only if it contains the gene 
    for sent in sentences_ls:
        if gene.lower() in sent.lower():
            pmid = sent.split(' ')[0]
            sent = sent.split(' ',1)[1]
            sent=re.sub(r'\b(%s)\b' % gene, r'<strong>\1</strong>', sent, flags=re.I)
            out+=pmid+"\t"+sent+"\n"
    return(out)

def gene_category(gene, cat_d, cat, abstracts,addiction_flag,dictn):
    # e.g. BDNF, addiction_d, undic(addiction_d) "addiction"
    sents=getSentences(gene, abstracts)
    out=str()
    if (addiction_flag==1):
        for sent in sents.split("\n"):
            for key in cat_d:
                if findWholeWord(cat_d[key])(sent) :
                    sent=sent.replace("<b>","").replace("</b>","") # remove other highlights
                    sent=re.sub(r'\b(%s)\b' % cat_d[key], r'<b>\1</b>', sent, flags=re.I) # highlight keyword
                    out+=gene+"\t"+ cat + "\t"+key+"\t"+sent+"\n"
    else:
        for sent in sents.split("\n"):
            for key_1 in dictn[cat_d].keys():
                for key_2 in dictn[cat_d][key_1]:
                    if findWholeWord(key_2)(sent) :
                        sent=sent.replace("<b>","").replace("</b>","") # remove other highlights
                        sent=re.sub(r'\b(%s)\b' % key_2, r'<b>\1</b>', sent, flags=re.I) # highlight keyword
                        out+=gene+"\t"+ cat + "\t"+key_1+"\t"+sent+"\n"
    return(out)

def generate_nodes(nodes_d, nodetype,nodecolor):
    # Include all search terms even if there are no edges, just to show negative result 
    json0 =str()
    for node in nodes_d:
        json0 += "{ data: { id: '" + node +  "', nodecolor: '" + nodecolor + "', nodetype: '"+nodetype + "', url:'/shownode?nodetype=" + nodetype + "&node="+node+"' } },\n"
    return(json0)

def generate_nodes_json(nodes_d, nodetype,nodecolor):
    # Include all search terms even if there are no edges, just to show negative result 
    nodes_json0 =str()
    for node in nodes_d:
        nodes_json0 += "{ \"id\": \"" + node +  "\", \"nodecolor\": \"" + nodecolor + "\", \"nodetype\": \"" + nodetype + "\", \"url\":\"/shownode?nodetype=" + nodetype + "&node="+node+"\" },\n"
    return(nodes_json0)

def generate_edges(data, filename):
    pmid_list=[]
    json0=str()
    edgeCnts={}

    for line in  data.split("\n"):
        if len(line.strip())!=0:
            (source, cat, target, pmid, sent) = line.split("\t")
            edgeID=filename+"|"+source+"|"+target
            if (edgeID in edgeCnts) and (pmid+target not in pmid_list):
                edgeCnts[edgeID]+=1
                pmid_list.append(pmid+target)
            elif (edgeID not in edgeCnts) and (pmid+target not in pmid_list):
                edgeCnts[edgeID]=1
                pmid_list.append(pmid+target)

    for edgeID in edgeCnts:
        (filename, source,target)=edgeID.split("|")
        json0+="{ data: { id: '" + edgeID + "', source: '" + source + "', target: '" + target + "', sentCnt: " + str(edgeCnts[edgeID]) + ",  url:'/sentences?edgeID=" + edgeID + "' } },\n"
    return(json0)

def generate_edges_json(data, filename):
    pmid_list=[]
    edges_json0=str()
    edgeCnts={}

    for line in  data.split("\n"):
        if len(line.strip())!=0:
            (source, cat, target, pmid, sent) = line.split("\t")
            edgeID=filename+"|"+source+"|"+target
            if (edgeID in edgeCnts) and (pmid+target not in pmid_list):
                edgeCnts[edgeID]+=1
                pmid_list.append(pmid+target)
            elif (edgeID not in edgeCnts) and (pmid+target not in pmid_list):
                edgeCnts[edgeID]=1
                pmid_list.append(pmid+target)

    for edgeID in edgeCnts:
        (filename, source,target)=edgeID.split("|")
        edges_json0+="{ \"id\": \"" + edgeID + "\", \"source\": \"" + source + "\", \"target\": \"" + target + "\", \"sentCnt\": \"" + str(edgeCnts[edgeID]) + "\",  \"url\":\"/sentences?edgeID=" + edgeID + "\" },\n"
    return(edges_json0)

def searchArchived(sets, query, filetype,sents, path_user):
    if sets=='topGene':
        dataFile="topGene_addiction_sentences.tab"
        nodes= "{ data: { id: '" + query +  "', nodecolor: '" + "#2471A3" + "', fontweight:700, url:'/progress?query="+query+"' } },\n"
    elif sets=='GWAS':
        dataFile="gwas_addiction.tab"
        nodes=str()   
    pmid_list=[]
    catCnt={}
    sn_file = ''
    
    for sn in sents:
        (symb, cat0, cat1, pmid, sent)=sn.split("\t")
        if (symb.upper() == query.upper()) :
            if (cat1 in catCnt.keys()) and (pmid+cat1 not in pmid_list):
                pmid_list.append(pmid+cat1)
                catCnt[cat1]+=1
            elif (cat1 not in catCnt.keys()):
                catCnt[cat1]=1
                pmid_list.append(pmid+cat1)
        sn_file += sn + '\n'

    nodes= "{ data: { id: '" + query +  "', nodecolor: '" + "#2471A3" + "', fontweight:700, url:'/progress?query="+query+"' } },\n"
    edges=str()
    gwas_json=str()
    nodecolor={}
    nodecolor["GWAS"]="hsl(0, 0%, 70%)"

    for key in catCnt.keys():
        if sets=='GWAS':
            nc=nodecolor["GWAS"]
            nodes += "{ data: { id: '" + key +  "', nodecolor: '" + nc + "', url:'https://www.ebi.ac.uk/gwas/search?query="+key.replace("_GWAS","")+"' } },\n"
        edgeID=path_user+'gwas_results.tab'+"|"+query+"|"+key
        edges+="{ data: { id: '" + edgeID+ "', source: '" + query + "', target: '" + key + "', sentCnt: " + str(catCnt[key]) + ",  url:'/sentences?edgeID=" + edgeID + "' } },\n"
        gwas_json+="{ \"id\": \"" + edgeID + "\", \"source\": \"" + query + "\", \"target\": \"" + key + "\", \"sentCnt\": \"" + str(catCnt[key]) + "\",  \"url\":\"/sentences?edgeID=" + edgeID + "\" },\n"
    return(nodes+edges,gwas_json,sn_file)

pubmed_path=os.environ["EDIRECT_PUBMED_MASTER"]

