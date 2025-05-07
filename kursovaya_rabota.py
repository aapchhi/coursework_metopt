'''import math
import random

def cal_dis(dis, path):
    length = 0
    for i in range(len(path) - 1):
        length += dis[path[i]][path[i + 1]]
    return length


def nearest_neighbor_heuristic(distance):
    path = [i for i in range(len(distance))]
    length = 0
    pre_node = random.choice(path)
    first_node = pre_node
    path.remove(pre_node)
    for i in range(len(distance) - 1):
        next_distance = [distance[pre_node][node] for node in path]
        cur_node = path[next_distance.index(min(next_distance))]
        path.remove(cur_node)
        length += distance[pre_node][cur_node]
        pre_node = cur_node
    length += distance[pre_node][first_node]
    return length

def roulette(pooling):
    sum_num = sum(pooling)
    temp_num = random.random()
    probability = 0
    for i in range(len(pooling)):
        probability += pooling[i] / sum_num
        if probability >= temp_num:
            return i
    return len(pooling)

def choose_next_city(dis, tau, beta, q0, ant_path):
    cur_city = ant_path[-1]
    roulette_pooling = []
    unvisited_cities = []
    for city in range(len(dis)):
        if city not in ant_path:
            unvisited_cities.append(city)
            roulette_pooling.append(tau[cur_city][city] * math.pow(1 / dis[cur_city][city], beta))
    if random.random() <= q0:  
        index = roulette_pooling.index(max(roulette_pooling))
    else:  
        index = roulette(roulette_pooling)
    return unvisited_cities[index]


def main(coord_x, coord_y, pop, iter, alpha, beta, rho, q0):

    city_num = len(coord_x)  
    dis = [[0 for _ in range(city_num)] for _ in range(city_num)]  
    for i in range(city_num):
        for j in range(i, city_num):
            temp_dis = math.sqrt((coord_x[i] - coord_x[j]) ** 2 + (coord_y[i] - coord_y[j]) ** 2)
            dis[i][j] = temp_dis
            dis[j][i] = temp_dis
    tau0 = 1 / (nearest_neighbor_heuristic(dis) * city_num)
    tau = [[tau0 for _ in range(city_num)] for _ in range(city_num)]  
    iter_best = []
    best_path = []
    best_length = 1e6
    start_city = [random.randint(0, city_num - 1) for _ in range(pop)] 

    
    for _ in range(iter):

        
        ant_path = [[start_city[i]] for i in range(pop)]
        for i in range(city_num):
            if i != city_num - 1:
                for j in range(pop):
                    next_city = choose_next_city(dis, tau, beta, q0, ant_path[j])
                    ant_path[j].append(next_city)
            else:
                for j in range(pop):
                    ant_path[j].append(start_city[j])

            
            #for j in range(pop):
             #   pre_city = ant_path[j][-2]
              #  cur_city = ant_path[j][-1]
               # temp_tau = (1 - rho) * tau[pre_city][cur_city] + rho * tau0
                #tau[pre_city][cur_city] = temp_tau
                #tau[cur_city][pre_city] = temp_tau


        for path in ant_path:
            length = cal_dis(dis, path)
            if length < best_length:
                best_length = length
                best_path = path
        iter_best.append(best_length)

        
        for i in range(city_num):
            for j in range(i, city_num):
                tau[i][j] *= (1 - alpha)
                tau[j][i] *= (1 - alpha)
        delta_tau = 1 / best_length
        for i in range(len(best_path) - 1):
            tau[best_path[i]][best_path[i + 1]] += delta_tau
            tau[best_path[i + 1]][best_path[i]] += delta_tau


        for i in range(city_num):
            for j in range(i, city_num):
                tau[i][j] *= (1 - alpha)
                tau[j][i] *= (1 - alpha)
        delta_tau = 1 / best_length
        for i in range(len(best_path) - 1):
            tau[best_path[i]][best_path[i + 1]] += delta_tau
            tau[best_path[i + 1]][best_path[i]] += delta_tau  

    x = [i for i in range(iter)]

    return {'Best tour': best_path, 'Shortest length': best_length}


if __name__ == '__main__':
    pop = 20
    beta = 2
    q0 = 0.9
    alpha = 0.1
    rho = 0.1
    iter = 50
    city_num = 30
    min_coord = 0
    max_coord = 10
    coord_x = [random.uniform(min_coord, max_coord) for _ in range(city_num)]
    coord_y = [random.uniform(min_coord, max_coord) for _ in range(city_num)]
    print(main(coord_x, coord_y, pop, iter, alpha, beta, rho, q0))
    '''


import math
import random
from matplotlib import pyplot as plt


class SolveTSPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight  
            self.pheromone = initial_pheromone  
            self.initial_pheromone = initial_pheromone  

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha  
            self.beta = beta  
            self.num_nodes = num_nodes  
            self.edges = edges  
            self.tour = None  
            self.distance = 0.0  

        def _select_node_roulette(self):
            roulette_wheel = 0.0  
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0  
           
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
         
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
           
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]  
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node_roulette())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, mode='Elitist', colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        self.distance_list = []

        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)

        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)

        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    """def _elitist(self):
            for step in range(self.steps):
            # Уменьшение феромонов на всех рёбрах
               for i in range(self.num_nodes):
                  for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

            for ant in self.ants:
                # Добавление феромонов на основе найденного тура
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance

            # Добавление феромонов на лучший найденный тур
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
            self.distance_list.append(self.global_best_distance)"""

    def _elitist(self):
        for step in range(self.steps):
            print(f"Step {step + 1}/{self.steps}:")
        
        # Уменьшение феромонов на всех рёбрах
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

            # Вывод значений феромонов после уменьшения
        print("  Pheromone levels after evaporation:")
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                print(f"    Edge ({i}, {j}): {self.edges[i][j].pheromone:.4f}")

            for ant in self.ants:
            # Добавление феромонов на основе найденного тура
                tour = ant.find_tour()
                distance = ant.get_distance()
                self._add_pheromone(tour, distance)
                print(f"  Ant {self.ants.index(ant)}: Tour = {tour}, Distance = {distance}")

                if distance < self.global_best_distance:
                   self.global_best_tour = ant.tour
                   self.global_best_distance = distance
                   print(f"  New global best distance: {self.global_best_distance}")
        # Добавление феромонов на лучший найденный тур
        self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
        self.distance_list.append(self.global_best_distance)
        print(f"  Global best tour: {self.global_best_tour}, Distance = {self.global_best_distance}\n")

        # Вывод значений феромонов после добавления
        print("  Pheromone levels after adding pheromones:")
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                print(f"    Edge ({i}, {j}): {self.edges[i][j].pheromone:.4f}")


    """def _max_min(self):
        for step in range(self.steps):
            iteration_best_tour = None
            iteration_best_distance = float("inf")

            for ant in self.ants:
                ant.find_tour()
                if ant.get_distance() < iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.distance

            # Уменьшение феромонов на всех рёбрах
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

            # Добавление феромонов на основе лучшего тура
            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_best_tour, iteration_best_distance)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
            else:
                if iteration_best_distance < self.global_best_distance:
                    self.global_best_tour = iteration_best_tour
                    self.global_best_distance = iteration_best_distance
                self._add_pheromone(self.global_best_tour, self.global_best_distance)
                max_pheromone = self.pheromone_deposit_weight / self.global_best_distance

            min_pheromone = max_pheromone * self.min_scaling_factor

            # Ограничение феромонов
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if self.edges[i][j].pheromone > max_pheromone:
                        self.edges[i][j].pheromone = max_pheromone
                    elif self.edges[i][j].pheromone < min_pheromone:
                        self.edges[i][j].pheromone = min_pheromone

            self.distance_list.append(self.global_best_distance)"""

    def _max_min(self):
      for step in range(self.steps):
        print(f"Step {step + 1}/{self.steps}:")
        iteration_best_tour = None
        iteration_best_distance = float("inf")

         # Вывод значений феромонов перед началом итерации
        print("  Pheromone levels before iteration:")
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                print(f"    Edge ({i}, {j}): {self.edges[i][j].pheromone:.4f}")

        for ant in self.ants:
            tour = ant.find_tour()
            distance = ant.get_distance()
            print(f"  Ant {self.ants.index(ant)}: Tour = {tour}, Distance = {distance}")

            if distance < iteration_best_distance:
                iteration_best_tour = tour
                iteration_best_distance = distance

        # Уменьшение феромонов на всех рёбрах
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j].pheromone *= (1.0 - self.rho)

                 # Вывод значений феромонов после уменьшения
        print("  Pheromone levels after evaporation:")
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                print(f"    Edge ({i}, {j}): {self.edges[i][j].pheromone:.4f}")

        # Добавление феромонов на основе лучшего тура
        if float(step + 1) / float(self.steps) <= 0.75:
            self._add_pheromone(iteration_best_tour, iteration_best_distance)
            max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
        else:
            if iteration_best_distance < self.global_best_distance:
                self.global_best_tour = iteration_best_tour
                self.global_best_distance = iteration_best_distance
                print(f"  New global best distance: {self.global_best_distance}")
            self._add_pheromone(self.global_best_tour, self.global_best_distance)
            max_pheromone = self.pheromone_deposit_weight / self.global_best_distance

        min_pheromone = max_pheromone * self.min_scaling_factor

        # Ограничение феромонов
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.edges[i][j].pheromone > max_pheromone:
                    self.edges[i][j].pheromone = max_pheromone
                elif self.edges[i][j].pheromone < min_pheromone:
                    self.edges[i][j].pheromone = min_pheromone

        self.distance_list.append(self.global_best_distance)
        print(f"  Global best distance after step: {self.global_best_distance}\n")

        # Вывод значений феромонов после добавления
        print("  Pheromone levels after adding pheromones:")
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                print(f"    Edge ({i}, {j}): {self.edges[i][j].pheromone:.4f}")


    def run(self):
        print('Started : {0}'.format(self.mode))
        if self.mode == 'Elitist':
            self._elitist()
        else:
            self._max_min()
        print('Ended : {0}'.format(self.mode))
        print('Sequence : {0}'.format(' -> '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))

    def plot_tour(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
        plt.title(self.mode)
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
        plt.show()
        plt.gcf().clear()

    def plot_opt(self):
        plt.figure(figsize=(12, 10))
        plt.plot(self.distance_list, label='Dist per step', color='b')
        plt.title('Train Curve')
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.show()


if __name__ == '__main__':
    _colony_size = 10
    _steps = 20
    random.seed(10)
    _nodes = [(random.uniform(0, 200), random.uniform(0, 200)) for _ in range(0, 10)]
    print(_nodes)

    elitist = SolveTSPUsingACO(mode='Elitist', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    elitist.run()
    elitist.plot_opt()
    elitist.plot_tour()

    max_min = SolveTSPUsingACO(mode='MaxMin', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    max_min.run()
    max_min.plot_opt()
    max_min.plot_tour()
