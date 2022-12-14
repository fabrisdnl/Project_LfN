import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

def get_mean_degrees(g):
    degrees = list(G.degree)
    mean_degree = 0
    mean_square_degree = 0

    for n in degrees:
        mean_degree += n[1]
        mean_square_degree += n[1] ** 2
    
    return [mean_degree / len(degrees), mean_square_degree / len(degrees)]

def get_sir_influent(g, betah, num_iter, num_output):
    
    start_time = time.time()
    beta = 1.5 * betah
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
    print("computation: %s seconds" % (time.time() - start_time))
    
    return [n[0] for n in sorted_d][:min(len(sorted_d), num_output)]

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

def print_influence(sets, labels):
    color = iter(cm.rainbow(np.linspace(0, 1, len(sets))))
    #s = ["SIR", "NLC", "Gravity", "LRASP"];
    for i in range(len(sets)):
       c = next(color)
       plt.plot(sets[i], c=c, label=labels[i])
    
    #TO DO: send sets as dictionary with key representing label of printed data
    plt.ylabel('num_infected')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()

def check_results(G, num, sets_indexes):
    mean_degree, mean_square_degree = get_mean_degrees(G)
    betah = mean_degree / mean_square_degree
    
    sir_output = get_sir_influent(G, betah, 1, num)
    sets = [sir_output[:num]]
    
    for s in sets_indexes:
        sets.append(s[:num])
    
    labels = = ["SIR", "NLC", "Gravity", "LRASP"];
    si_output = compute_sets_si(sets, betah, 30)
    print_influence(si_output, labels[:len(si_output)])
