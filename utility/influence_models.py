import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot
import numpy as np

def get_mean_degrees(g):
    degrees = list(G.degree)
    mean_degree = 0
    mean_square_degree = 0

    for n in degrees:
        mean_degree += n[1]
        mean_square_degree += n[1] ** 2
    
    return [mean_degree / len(degrees), mean_square_degree / len(degrees)]

def get_sir_influent(g, num_iter, num_output):
    md, msd = get_mean_degrees(g)
    
    beta = 1.5 * md / msd
    gamma = 1
    d = dict()
    for n in nodes:
        d[n] = 0

    # Model Configuration
    for i in range(num_iter):
        for n in g.nodes:
            status = True
            model = ep.SIRModel(g)
            cfg = mc.Configuration()
            cfg.add_model_parameter('beta', beta)
            cfg.add_model_parameter('gamma', gamma)
            cfg.add_model_initial_configuration("Infected", [n])
            model.set_initial_status(cfg)

            while(status):
                iterations = model.iteration()
                if iterations['node_count'][1] <= 0:
                    status = False

            d[n] += iterations['node_count'][2]
            
    for n in nodes:
        d[n] = d[n] / num_iter

    sorted_d = sorted(d.items(), key=lambda kv: kv[1])
    sorted_d.reverse()
    
    return sorted_d[:min(len(sorted_d), number)]

def compute_sets_si(set_list, betah, interval):
    beta = 2 * betah
    results = []
    
    for l in set_list:
        current = []
        modelSI = ep.SIModel(G)
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', beta)
        cfg.add_model_initial_configuration("Infected", l)
        modelSI.set_initial_status(cfg)
        
        for i in range(interval):
            iterations = modelSI.iteration()
            current.append(iterations['node_count'][1])
        
        results.append(current)
  
    return results

def print_influence(sets):
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    
    for y in sets:
       c = next(color)
       plt.plot(y, c=c)
    
    #TO DO: send sets as dictionary with key representing label of printed data
    plt.ylabel('num_infected')
    plt.xlabel('iterations')
    plt.show()
