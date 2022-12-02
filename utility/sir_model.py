import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

def get_mean_degrees(g):
    degrees = list(G.degree)
    mean_degree = 0
    mean_square_degree = 0

    for n in degrees:
        mean_degree += n[1]
        mean_square_degree += n[1] ** 2
    
    return [mean_degree / len(degrees), mean_square_degree / len(degrees)]

def get_most_influent(g, number):
    
    md, msd = get_mean_degrees(g)
    
    d = dict()
    beta = 1.5 * md / msd
    gamma = 1

    # Model Configuration
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

        d[n] = iterations['node_count'][2]

    sorted_d = sorted(d.items(), key=lambda kv: kv[1])
    sorted_d.reverse()
    
    return sorted_d[0:min(len(sorted_d), number)]