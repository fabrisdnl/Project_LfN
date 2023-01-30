import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from matplotlib.pyplot import cm
import time

class Infection:
    @classmethod
    def get_mean_degrees(self, g):
        degrees = list(g.degree)
        mean_degree = 0
        mean_square_degree = 0

        for n in degrees:
            mean_degree += n[1]
            mean_square_degree += n[1] ** 2

        return [mean_degree / len(degrees), mean_square_degree / len(degrees)]

    @classmethod
    def get_sir_influent(self, g, betah, iter_exec, num_iter, num_output):

        start_time = time.time()
        beta = 1.5 * betah
        gamma = 1
        d = dict()
        mapping = dict(zip(g, range(1, len(list(g.nodes())))))
        g = nx.relabel_nodes(g, mapping)
        for n in g.nodes:
            d[n] = 0

        # Model Configuration
        for i in range(num_iter):
            for n in g.nodes:
                status = True
                count = 0
                model = ep.SIRModel(g)
                cfg = mc.Configuration()
                cfg.add_model_parameter('beta', beta)
                cfg.add_model_parameter('gamma', gamma)
                cfg.add_model_initial_configuration("Infected", [n])
                model.set_initial_status(cfg)

                while(count < iter_exec):
                    iterations = model.iteration()
                    count = count + 1
                    if iterations['node_count'][1] <= 0:
                        status = False

                d[n] += iterations['node_count'][2] + iterations['node_count'][1]

            print("iteration " + str(i) + " concluded", end="\r", flush=True)

        for n in g.nodes:
            d[n] = d[n] / num_iter

        sorted_d = sorted(d.items(), key=lambda kv: kv[1])
        sorted_d.reverse()
        print("computation: %s seconds" % (time.time() - start_time))

        return [n[0] for n in sorted_d][:min(len(sorted_d), num_output)]

    @classmethod
    def compute_sets_si(self, set_list, betah, interval, iter_count):
        beta = 2 * betah
        results = []

        for l in set_list:
            current = [0]*interval

            for it in range(iter_count):
                modelSI = ep.SIModel(G)
                cfg = mc.Configuration()
                cfg.add_model_parameter('beta', beta)
                cfg.add_model_initial_configuration("Infected", l)
                modelSI.set_initial_status(cfg)

                for i in range(interval):
                    iterations = modelSI.iteration()
                    current[i] = current[i] + iterations['node_count'][1]

            for i in range(len(current)):
                current[i] = current[i] / iter_count

            results.append(current)

        return results

    @classmethod
    def print_influence(self, sets, labels, title):
        color = iter(cm.rainbow(np.linspace(0, 1, len(sets))))
        
        for i in range(len(sets)):
            c = next(color)
            plt.plot(sets[i], c=c, label=labels[i])
            plt.plot(sets[i], '.', c=c)

        
        plt.title(title)
        plt.ylabel('num_infected')
        plt.xlabel('iterations')
        plt.grid(True)
        plt.legend()
        plt.show()

    def get_ground(self, G, f_value, num_output, num_iter):
        mean_degree, mean_square_degree = self.get_mean_degrees(G)
        betah = mean_degree / mean_square_degree

        print("Computing SIR results")
        sir_output = self.get_sir_influent(G, betah, f_value, num_iter, num_output)
        print(sir_output)

        return sir_output