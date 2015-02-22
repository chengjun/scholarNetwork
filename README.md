#scholarNetwork

Developed by Cheng-Jun Wang & Lingfei Wu

Cheng-Jun Wang wangchj04@gmail.com
Lingfei Wu wlf850927@gmail.com

## Introduction

scholarNetwork is a python package for crawl and visualize the co-author network of Google Scholar.

To use it, you must have access to Google Scholar, and you would also install beautifulsoup4 and networkx during the installation process.

## Install
Install from pypi using pip or easy_install

	pip install scholarNetwork

or

	easy_install scholarNetwork

##Use
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


![](https://github.com/chengjun/scholarNetwork/blob/master/example.png)