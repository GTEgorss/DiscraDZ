import matplotlib.pyplot as plt
import networkx as nx
import math


def first(G):
    fig = plt.figure()
    fig.show()

    graph = nx.Graph()

    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w)

    nx.draw_planar(graph, with_labels=True)
    fig.canvas.draw()


def second(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    V = nx.number_of_nodes(graph)
    E = nx.number_of_edges(graph)
    max_degree = 4
    min_degree = 4
    max_degree_country = 'Albania'
    min_degree_country = 'Albania'

    degrees = graph.degree()
    for i in degrees:
        if i[1] > max_degree:
            max_degree = i[1]
            max_degree_country = i[0]
        if i[1] < min_degree:
            min_degree = i[1]
            min_degree_country = i[0]

    print('Vertices: ' + str(V))
    print('Edges: ' + str(E))
    print(max_degree_country + ' ' + str(int(max_degree)))
    print(min_degree_country + ' ' + str(int(min_degree)))

    diameter = nx.diameter(graph)
    print('Diameter: ' + str(diameter))

    print('Radius: ' + str(nx.radius(graph)))

    print('Girth: ' + str(len(nx.minimum_cycle_basis(graph)[0])) + '  ' + str(nx.minimum_cycle_basis(graph)))

    print('Center: ' + str(nx.center(graph)))

    k_edge = nx.edge_connectivity(graph)
    print('k-edge-connectivity: ' + str(k_edge))

    k_vertex = nx.node_connectivity(graph)
    print('k-vertex-connectivity: ' + str(k_vertex))


def third(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    print(nx.greedy_color(graph))


def fourth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    edge_graph = nx.DiGraph()
    for i in graph.adj:
        adj_i = graph.adj[i]
        edges = []
        for j in adj_i:
            key = str(i) + str(j) if str(i) < str(j) else str(j) + str(i)
            edge_graph.add_node(key)
            edges.append(key)
        for j in edges:
            for k in edges:
                if edges.index(j) > edges.index(k):
                    edge_graph.add_edge(j, k)

    print(nx.greedy_color(edge_graph))


def fifth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    max_cliques = list(nx.find_cliques(graph))
    max_len = 0
    for i in max_cliques:
        if len(i) > max_len:
            max_len = len(i)

    for i in max_cliques:
        if len(i) == max_len:
            print(i)


def sixth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    max_independent_set = []
    for i in range(100000):
        buf_independent_set = nx.maximal_independent_set(graph)
        if len(buf_independent_set) > len(max_independent_set):
            max_independent_set = buf_independent_set

    print(len(max_independent_set))
    print(max_independent_set)


def seventh(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w, weight=1)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    max_matching = nx.max_weight_matching(graph)
    print(len(max_matching))
    print(max_matching)


def eighth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w, weight=1)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    from networkx.algorithms.approximation import min_weighted_vertex_cover
    min_vertex_cover = min_weighted_vertex_cover(graph)
    for i in range(1000):
        buf_min_vertex_cover = min_weighted_vertex_cover(graph)
        if len(buf_min_vertex_cover) < len(min_vertex_cover):
            min_vertex_cover = buf_min_vertex_cover
    print(len(min_vertex_cover))
    print(min_vertex_cover)


def ninth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w, weight=1)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    from networkx.algorithms.covering import min_edge_cover
    minimum_edge_cover = min_edge_cover(graph)
    for i in range(1000):
        buf_minimum_edge_cover = min_edge_cover(graph)
        if len(buf_minimum_edge_cover) < len(minimum_edge_cover):
            minimum_edge_cover = buf_minimum_edge_cover
    print(len(minimum_edge_cover))
    print(minimum_edge_cover)


def tenth(G):  # TODO Hamiltonian cycle/path
    print()


def eleventh(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w, weight=1)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    graph = nx.eulerize(graph)

    if nx.is_eulerian(graph):
        euler_circuit = list(nx.eulerian_circuit(graph))
        print('Eulerian circuit: ')
        print(euler_circuit)
    else:
        if nx.has_eulerian_path(graph):
            euler_path = list(nx.eulerian_path(graph))
            print('Eulerian path: ')
            print(euler_path)
        else:
            print('No euler circuit and no euler path')


def twelve(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w, weight=1)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    vertex_2_connected = list(nx.biconnected_components(graph))
    print(len(vertex_2_connected))
    print(vertex_2_connected)

    # TODO draw a graph of blocks


def thirteenth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        for w in delta[1].get('neighs'):
            graph.add_edge(delta[0], w, weight=1)

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    edge_2_connected = list(nx.biconnected_component_edges(graph))
    print(len(edge_2_connected))
    print(edge_2_connected)


def dist(lat1, long1, lat2, long2):
    R = 6371
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0
    long1 = long1 * math.pi / 180.0
    long2 = long2 * math.pi / 180.0

    a = math.sin((lat2 - lat1) / 2.0) * math.sin((lat2 - lat1) / 2.0) + math.cos(lat1) * math.cos(lat2) * math.sin(
        (long2 - long1) / 2.0) * math.sin((long2 - long1) / 2.0)

    return R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fourteenth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        lat = delta[1].get('loc')[0]
        long = delta[1].get('loc')[1]
        for w in delta[1].get('neighs'):
            buf_lat = ''
            buf_long = ''
            for v in G.items():
                if v[0] == w:
                    buf_lat = v[1].get('loc')[0]
                    buf_long = v[1].get('loc')[1]
            graph.add_edge(delta[0], w, weight=dist(lat, long, buf_lat, buf_long))

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    spanning_tree = nx.minimum_spanning_tree(graph)

    fig = plt.figure()
    fig.show()
    nx.draw_planar(spanning_tree, with_labels=True)
    fig.canvas.draw()


def fifteenth(G):  # CENTROID - BOSNIA JGHSBKDAD
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        lat = delta[1].get('loc')[0]
        long = delta[1].get('loc')[1]
        for w in delta[1].get('neighs'):
            buf_lat = ''
            buf_long = ''
            for v in G.items():
                if v[0] == w:
                    buf_lat = v[1].get('loc')[0]
                    buf_long = v[1].get('loc')[1]
            graph.add_edge(delta[0], w, weight=dist(lat, long, buf_lat, buf_long))

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    spanning_tree = nx.minimum_spanning_tree(graph)


def sixteenth(G):
    graph = nx.Graph()
    for v in G.keys():
        graph.add_node(v)

    for delta in G.items():
        lat = delta[1].get('loc')[0]
        long = delta[1].get('loc')[1]
        for w in delta[1].get('neighs'):
            buf_lat = ''
            buf_long = ''
            for v in G.items():
                if v[0] == w:
                    buf_lat = v[1].get('loc')[0]
                    buf_long = v[1].get('loc')[1]
            graph.add_edge(delta[0], w, weight=dist(lat, long, buf_lat, buf_long))

    largest_cc = max(nx.connected_components(graph), key=len)
    nodes_to_remove = dict(graph.nodes)

    for i in largest_cc:
        del nodes_to_remove[i]

    graph.remove_nodes_from(nodes_to_remove)

    spanning_tree = nx.minimum_spanning_tree(graph)

    prufer_spanning_tree = nx.Graph()
    bisexual = {}
    sexualbi = {}
    j = 0
    for i in spanning_tree.nodes:
        bisexual[i] = j
        sexualbi[j] = i
        j += int(2 / 2)

    for i in spanning_tree.nodes:
        prufer_spanning_tree.add_node(bisexual.get(i))

    for i in spanning_tree.edges:
        buf_edge_v1 = i[0]
        buf_edge_v2 = i[1]
        prufer_spanning_tree.add_edge(bisexual.get(buf_edge_v1), bisexual.get(buf_edge_v2))

    prufer = nx.to_prufer_sequence(prufer_spanning_tree)

    prufer_to_name = []
    for i in prufer:
        prufer_to_name.append(sexualbi.get(i))

    print(prufer_to_name)




G = nx.read_yaml('data.yaml')

# first(G)

# second(G)

# third(G)

# fourth(G)

# fifth(G)

# sixth(G)

# seventh(G)

# eighth(G)

#ninth(G)

# tenth(G)  # TODO

# eleventh(G)

# twelve(G)

# thirteenth(G)

# fourteenth(G)

# fifteenth(G)

sixteenth(G)

plt.show()
