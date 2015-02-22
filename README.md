# Install
Install from pypi using pip or easy_install

	pip install scholarNetwork

or

	easy_install scholarNetwork

# Use
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

