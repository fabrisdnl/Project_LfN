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
	it = 0
    core_values = dict()
    h = G.copy()
    
    if len(list(nx.selfloop_edges(h))) != 0:
        h.remove_edges_from(nx.selfloop_edges(h))
        
    while(1):
        flag = check(h, it)
        if (flag == 0):
            #print("bucket " + str(it) + " added")
            it += 1
        if (flag == 1):
            node_set = find_nodes(h, it)
            for each in node_set:
                h.remove_node(each)
                core_values[each] = it
        if (h.number_of_nodes() == 0):
            break

    return core_values
