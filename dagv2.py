from copy import deepcopy
from collections import deque, OrderedDict

class DAGValidationError(Exception):
    pass


class DAG(object):
    """ Directed acyclic graph implementation. """

    def __init__(self):
        """ Construct a new DAG with no nodes or edges. """
        self.graph = OrderedDict()

    def add_node(self, node_name, conv=None):
        """ Add a node with an optional conv if it does not exist yet, or error out. """
        if node_name in self.graph:
            raise KeyError(f'node {node_name} allready exists')
        
        self.graph[node_name] = {"edges": set(), 
                                 "conv": conv}
        
    def add_conv(self, node_name, conv):
        if node_name not in self.graph:
            raise KeyError(f'node {node_name} not exists')
        
        self.graph[node_name]['conv'] = conv
    
    def add_node_if_not_exists(self, node_name, conv=None):
        try:
            self.add_node(node_name, conv)
        except KeyError:
            pass

    def delete_node(self, node_name):
        """ Deletes this node and all edges referencing it. """
        if node_name not in self.graph:
            raise KeyError(f'node {node_name} does not exist')
        
        self.graph.pop(node_name)

        for _, data in self.graph.items():
            if node_name in data['edges']:
                data['edges'].remove(node_name)

    def delete_node_if_exists(self, node_name):
        try:
            self.delete_node(node_name)
        except KeyError:
            pass

    def ind_node(self, graph=None):
        """ Returns a list of all nodes in the graph with no dependencies. """
        if graph is None: graph = self.graph

        dependent_nodes = [node for data in graph.values() for node in data['edges']]

        return [node for node in graph.keys() if node not in dependent_nodes]

    def validate(self, graph=None):
        """ Returns (Boolean, message) of whether DAG is valid. """
        graph = graph if graph is not None else self.graph

        if len(self.ind_node(graph)) == 0:
            return (False, 'no independent nodes detected')
        
        try:
            self.topology_sort(graph)
        except ValueError:
            return (False, 'failed topological sort')
        
        return (True, 'valid')
    
    def topology_sort(self, graph=None):
        """ Returns a topological ordering of the DAG.
        Raises an error if this is not possible (graph is not valid).
        Using Kahn algorithm """

        if graph is None: graph = self.graph

        in_degree = {u: 0 for u in graph}
        for u in graph:
            for v in graph[u]['edges']:
                in_degree[v] += 1

        queue = deque([u for u in in_degree if in_degree[u] == 0])

        l = []
        while queue:
            u = queue.pop()
            l.append(u)
            for v in graph[u]['edges']:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.appendleft(v)

        if len(l) == len(graph):
            return l
        
        raise ValueError('graph is not acyclic')

    def size(self):
        return len(self.graph)

    def add_edge(self, ind_node, dep_node):
        """ Add an edge (dependency) between the specified nodes. """
        if ind_node not in self.graph or dep_node not in self.graph:
            raise KeyError('one or more nodes do not exist in graph')
        
        test_graph = deepcopy(self.graph)
        test_graph[ind_node]['edges'].add(dep_node)
        is_valid, message = self.validate(test_graph)

        del test_graph

        if is_valid:
            self.graph[ind_node]['edges'].add(dep_node)
        else:
            raise ValueError(message)
        

    def delete_edge(self, ind_node, dep_node):
        """ Delete an edge from the graph. """
        if dep_node not in self.graph.get(ind_node, {'edges': set()})['edges']:
            raise KeyError('this edge does not exist in graph')
        
        self.graph[ind_node]['edges'].remove(dep_node)

    def rename_edges(self, old_task_name, new_task_name):
        """ Change references to a task in existing edges. """
        for node, data in self.graph.items():
            if node == old_task_name:
                self.graph[new_task_name] = data
                del self.graph[old_task_name]
            else:
                if old_task_name in data['edges']:
                    data['edges'].remove(old_task_name)
                    data['edges'].add(new_task_name)

    def predecessors(self, node):
        """ Returns a list of all predecessors of the given node """
        return [key for key, data in self.graph.items() if node in data['edges']]
    
    def downstream(self, node):
        """ Returns a list of all nodes this node has edges towards. """
        if node not in self.graph:
            raise KeyError(f'node {node} is not in graph')
        
        return list(self.graph[node]['edges'])
    
    def all_downstreams(self, node):
        """ Returns a list of all nodes ultimately downstream
        of the given node in the dependency graph, in
        topological order. """
        nodes = [node]
        nodes_seen = set()
        i = 0

        while i < len(nodes):
            downstreams = self.downstream(node[i])
            for downstream_node in downstreams:
                if downstream_node not in nodes_seen:
                    nodes_seen.add(downstream_node)
                    nodes.append(downstream_node)

            i += 1

        return list(filter(lambda node: node in nodes_seen, self.topology_sort()))
    
    def all_leaves(self):
        """ Return a list of all leaves (nodes with no downstreams) """
        return [key for key, data in self.graph.items() if not data['edges']]
    
    def from_dict(self, graph_dict):
        """ Reset the graph and build it from the passed dictionary.
        The dictionary takes the form of {node_name: [directed edges]} """
        self.graph = OrderedDict()

        for new_node in graph_dict.keys():
            self.add_node(new_node)

        for ind_node, dep_nodes in graph_dict.items():
            if not isinstance(dep_node, list):
                raise TypeError('dict values must be lists')
            for dep_node in dep_nodes:
                self.add_edge(ind_node, dep_node)

    def get_conv(self, node_name):
        """ Return the conv associated with a node """
        if node_name in self.graph:
            return self.graph[node_name]['conv']
        else:
            raise KeyError(f'node {node_name} does not exist')


# dag = DAG()
# dag.add_node('node1', tf.constant([1, 2, 3]))
# dag.add_node('node2', tf.constant([4, 5, 6]))
# dag.add_edge('node1', 'node2')

# print(dag.get_conv('node1'))  # Output: tf.conv([1 2 3], shape=(3,), dtype=int32)
# print(dag.get_conv('node2'))  # Output: tf.conv([4 5 6], shape=(3,), dtype=int32)