import networkx as nx

def check(h, d):
    f = 0
    for i in h.nodes():
        if(h.degree(i) <= d):
            f = 1
            break
    return f

def find_nodes(h, it):
    s = []
    for i in h.nodes():
        if(h.degree(i) <= it):
            s.append(i)
    return s
	
def kShell_values(h):
	it = 1
	tmp = []
	buckets = []

	while(1):
		flag = check(h, it)
		if (flag == 0):
			it += 1
			buckets.append(tmp)
			tmp = []
		if (flag == 1):
			node_set = find_nodes(h, it)
			for each in node_set:
				h.remove_node(each)
				tmp.append(each)
		if (h.number_of_nodes() == 0):
			buckets.append(tmp)
			break

	core_values = dict()

	value = 1

	for b in buckets:
		for n in b:
			core_values[n] = value
		value += 1
		
	return core_values