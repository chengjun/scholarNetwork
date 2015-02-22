# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 14:19:41 2015

@author: chengjun
"""

import urllib2
from bs4 import BeautifulSoup
from collections import defaultdict
import networkx as nx



def getGraph(seed, Nmax):
    urls = defaultdict(int)
    urls[seed]+=1
    newUrls = [seed]# initiate the coauthor list
    G = nx.DiGraph()
    def coAuthors(url): 
        coUrls = []
        coNames = [] #for network plot
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html)
        s = soup.body.findAll('a', {"class": "gsc_rsb_aa"})
        egoName = soup.body.find('div', {"id": "gsc_prf_in"}).text #for network plot
        if s:
            for i in s:
                coNames.append(i.text) #for network plot
                coUrls.append('http://scholar.google.nl'+ i['href'])
        for j in coUrls:
            urls[j] += 0 
        for m in coNames: #for network plot
            G.add_edge(egoName.split(',')[0], m.split(',')[0])
        return coUrls
    
    while newUrls:
        for k in urls.keys(): # update url.values() first
            urls[k] += 1 
        addUrls = [] # get new-added authors, may have duplications.
        for i in newUrls:
            coUrls = coAuthors(i)
            if coUrls:
                for j in coUrls:
                    addUrls.append(j)
        for m in set(addUrls): # get rid of the duplications
            urls[m] += 1
        newUrls = [k for k, v in urls.items() if v <= 1]# update the new coauthors and avoid the deadloop: a->b->a->......
        addUrls = []   
        print len(urls.keys())
        if len(urls.keys()) > Nmax:
            print 'more than '+str(Nmax)+' people now, break'
            break
    return G

    
### plot local network
#import matplotlib.pyplot as plt
#seed = 'https://scholar.google.nl/citations?user=nNdt_G8AAAAJ&hl=en&oe=ASCII'
#Nmax = 500
#g = getGraph(seed, Nmax)
#pos=nx.spring_layout(G) #设置网络的布局
#fig = plt.figure(figsize=(40, 40),facecolor='white')
#nx.draw(G, pos, node_shape = 'o',
#        edge_color = 'gray', width = 0.5,
#        with_labels = True, arrows = True)
#plt.show()