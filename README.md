#scholarNetwork

Developed by Cheng-Jun Wang & Lingfei Wu

Cheng-Jun Wang wangchj04@gmail.com
Lingfei Wu wlf850927@gmail.com

## Introduction

scholarNetwork is a python package for crawling and visualizing the co-author network of Google Scholar.

To use it, you must have access to Google Scholar, and you would also install beautifulsoup4 and networkx during the installation process.

## Install
Install from pypi using pip or easy_install

	pip install scholarNetwork

or

	easy_install scholarNetwork

##Use

```python
# scholarNetwork

from scholarNetwork import scholarNetwork
import matplotlib.pyplot as plt
import networkx as nx

## The seed of crawler
seed = 'https://scholar.google.nl/citations?user=nNdt_G8AAAAJ&hl=en&oe=ASCII'
# How many nodes do you want to visulize? Always start with a small one. 
Nmax = 21
## Get the graph g
g = scholarNetwork.getGraph(seed, Nmax)

## plot the network
pos=nx.spring_layout(g) #setup the layout

nx.draw(g, pos, node_shape = 'o',
	edge_color = 'gray', width = 0.5,
	with_labels = True, arrows = True)
plt.show()
```

![](https://github.com/chengjun/scholarNetwork/blob/master/example.png)

Of course, you can get a much larger graph to see what's happening around you, just to change the value of Nmax. However, you have to be patient to wait for much longer time.

![](http://chengjun.qiniudn.com/ego300large.png)

## Plot tree graph

```python
G=nx.DiGraph()
G.add_edge('source',1,weight=80)
G.add_edge(1,2,weight=50)
G.add_edge(1,3,weight=30)
G.add_edge(3,2,weight=10)
G.add_edge(2,4,weight=20)
G.add_edge(2,5,weight=30)
G.add_edge(4,5,weight=10)
G.add_edge(5,3,weight=5)
G.add_edge(2,'sink',weight=10)
G.add_edge(4,'sink',weight=10)
G.add_edge(3,'sink',weight=25)
G.add_edge(5,'sink',weight=35)

fig = plt.figure(figsize=(9, 9),facecolor='white')
ax = fig.add_subplot(111)
scholarNetwork.plotTree(G,ax)
plt.show()
```
For the coauthor network we have crawled, we can also visualize it as a tree. 

```python
fig = plt.figure(figsize=(9, 9),facecolor='white')
ax = fig.add_subplot(111)
scholarNetwork.plotTree(g,ax)
plt.show()
```
![](https://github.com/chengjun/scholarNetwork/blob/master/tree.png)
