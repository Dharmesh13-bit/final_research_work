import numpy as np
import tensorflow as tf
import networkx as nx
import random
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =========================
# LOAD MODELS
# =========================

NUM_NODES = 50

ThreNodes = 8
ThreRes = 40
ThreLat = 6

# =========================
# NETWORK GENERATION
# =========================
def generate_iot_graph(n=NUM_NODES):
    while True:
        G = nx.waxman_graph(n=n, alpha=0.5, beta=0.5)
        if nx.is_connected(G):
            break

    for node in G.nodes():
        G.nodes[node]['cpu'] = random.randint(80, 100)
        G.nodes[node]['storage'] = random.randint(80, 100)
        G.nodes[node]['delay'] = random.randint(1, 3)

    for u, v in G.edges():
        G[u][v]['bandwidth'] = random.randint(80, 100)
        G[u][v]['delay'] = random.randint(1, 3)

    return G

# =========================
# SERVICE GENERATION
# =========================
def generate_service():
    while True:
        n = random.randint(5, 10)
        S = nx.erdos_renyi_graph(n, 0.6)
        if nx.is_connected(S):
            break

    for node in S.nodes():
        S.nodes[node]['cpu_req'] = random.randint(10, 20)
        S.nodes[node]['storage_req'] = random.randint(10, 20)
        S.nodes[node]['delay_req'] = random.randint(2, 5)

    for u, v in S.edges():
        S[u][v]['bandwidth_req'] = random.randint(10, 20)
        S[u][v]['delay_req'] = random.randint(2, 5)

    return S

# =========================
# SERVICE CLASSIFICATION
# =========================
def classify_service(service):
    num_nodes = service.number_of_nodes()

    avg_res = sum(service.nodes[n]['cpu_req'] + service.nodes[n]['storage_req']
                  for n in service.nodes()) / num_nodes

    avg_bw = (sum(service[u][v]['bandwidth_req']
                 for u, v in service.edges()) / service.number_of_edges()
              if service.number_of_edges() > 0 else 0)

    avg_res += avg_bw

    avg_lat = sum(service.nodes[n]['delay_req']
                  for n in service.nodes()) / num_nodes

    avg_link_lat = (sum(service[u][v]['delay_req']
                       for u, v in service.edges()) / service.number_of_edges()
                    if service.number_of_edges() > 0 else 0)

    avg_lat += avg_link_lat

    if num_nodes >= ThreNodes:
        return "multi"
    elif avg_res >= ThreRes:
        return "high"
    elif avg_lat <= ThreLat:
        return "low"
    else:
        return "random"

# =========================
# DATASET
# =========================
def generate_dataset():
    high, low, multi = [], [], []

    while len(high) < 1000 or len(low) < 1000 or len(multi) < 1000:
        s = generate_service()
        c = classify_service(s)

        if c == "high" and len(high) < 1000:
            high.append(s)
        elif c == "low" and len(low) < 1000:
            low.append(s)
        elif c == "multi" and len(multi) < 1000:
            multi.append(s)

    print("Dataset Ready:", len(high), len(low), len(multi))
    return high, low, multi

# =========================
# GNN
# =========================
class GNNLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(out_dim)

    def call(self, X, A):
        I = tf.eye(A.shape[0])
        A_hat = A + I
        D = tf.reduce_sum(A_hat, axis=1)
        D_inv = tf.linalg.diag(1.0 / tf.sqrt(D + 1e-8))
        A_norm = tf.matmul(tf.matmul(D_inv, A_hat), D_inv)
        return tf.nn.relu(self.dense(tf.matmul(A_norm, X)))

class GNNEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.g1 = GNNLayer(32)
        self.g2 = GNNLayer(32)

    def call(self, X, A):
        return self.g2(self.g1(X, A), A)

# =========================
# POLICY NETWORK
# =========================
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(1)

    def call(self, emb):
        x = self.d1(emb)
        x = self.d2(x)
        x = tf.squeeze(x, axis=-1)
        return tf.nn.softmax(x)
    
def load_model(weight_file):
    gnn = GNNEncoder()
    policy = PolicyNetwork()

    # build models
    dummy_X = tf.random.normal((NUM_NODES, 3))
    dummy_A = tf.random.normal((NUM_NODES, NUM_NODES))
    _ = gnn(dummy_X, dummy_A)
    _ = policy(tf.random.normal((NUM_NODES, 32)))

    policy.load_weights(weight_file)

    return gnn, policy

# =========================
# METRICS
# =========================

def generate_iot_graph(n=NUM_NODES):
    while True:
        G = nx.waxman_graph(n=n, alpha=0.5, beta=0.5)
        if nx.is_connected(G):
            break

    for node in G.nodes():
        G.nodes[node]['cpu'] = random.randint(80, 100)
        G.nodes[node]['storage'] = random.randint(80, 100)
        G.nodes[node]['delay'] = random.randint(1, 3)

    for u, v in G.edges():
        G[u][v]['bandwidth'] = random.randint(80, 100)
        G[u][v]['delay'] = random.randint(1, 3)

    return G

class Metrics:
    def __init__(self):
        self.success = 0
        self.total = 0

        self.cpu_used = 0
        self.storage_used = 0
        self.bandwidth_used = 0

        self.total_revenue = 0
        self.total_cost = 0

        # ✅ NEW (correct utilization tracking)
        self.cpu_util_total = 0
        self.storage_util_total = 0
        self.bandwidth_util_total = 0


# =========================
# TEST FUNCTION (FIXED)
# =========================

def build_features(G):
    nodes = list(G.nodes())
    X = [[G.nodes[n]['cpu'], G.nodes[n]['storage'], G.nodes[n]['delay']] for n in nodes]
    return np.array(X, dtype=np.float32), nodes

def build_adj(G, nodes):
    N = len(nodes)
    A = np.zeros((N, N))
    idx = {n:i for i,n in enumerate(nodes)}
    for u,v in G.edges():
        A[idx[u]][idx[v]] = 1
        A[idx[v]][idx[u]] = 1
    return A.astype(np.float32)

def find_path(G, u, v, bw, delay):
    try:
        path = nx.shortest_path(G, u, v, weight='delay')
    except:
        return None

    total = 0
    for i in range(len(path)-1):
        e = G[path[i]][path[i+1]]
        if e['bandwidth'] < bw:
            return None
        total += e['delay']

    return path if total <= delay else None



def test_model(services, weight_file, name):

    print(f"\n🧪 Testing {name}")
    gnn, policy = load_model(weight_file)
    metrics = Metrics()
    G = generate_iot_graph()
    
    for service in services:
        G_copy = copy.deepcopy(G)         
        

        # ✅ total resources for THIS graph only
        total_cpu = sum(G_copy.nodes[n]['cpu'] for n in G_copy.nodes())
        total_storage = sum(G_copy.nodes[n]['storage'] for n in G_copy.nodes())
        total_bandwidth = sum(G_copy[u][v]['bandwidth'] for u, v in G_copy.edges())

        # track used per service
        cpu_used_service = 0
        storage_used_service = 0
        bandwidth_used_service = 0

        X, nodes = build_features(G_copy)
        A = build_adj(G_copy, nodes)

        emb = gnn(X, A)
        probs = policy(emb).numpy()

        mapping = {}
        success = True

        # ======================
        # NODE ALLOCATION
        # ======================
        def can_allocate(node, demand):
            return node['cpu'] >= demand['cpu_req'] and \
                   node['storage'] >= demand['storage_req'] and \
                   node['delay'] <= demand['delay_req']

        for s in service.nodes():

            demand = service.nodes[s]

            mask = np.array([
                1 if can_allocate(G_copy.nodes[n], demand) else 0
                for n in nodes
            ])

            masked = probs * mask

            if masked.sum() == 0:
                success = False
                break

            masked = masked / masked.sum()
            idx = np.argmax(masked)

            mapping[s] = nodes[idx]

            G_copy.nodes[nodes[idx]]['cpu'] -= demand['cpu_req']
            G_copy.nodes[nodes[idx]]['storage'] -= demand['storage_req']

            cpu_used_service += demand['cpu_req']
            storage_used_service += demand['storage_req']

        # ======================
        # LINK ALLOCATION
        # ======================
        paths = {}

        if success:
            for u, v in service.edges():

                p = find_path(G_copy, mapping[u], mapping[v],
                              service[u][v]['bandwidth_req'],
                              service[u][v]['delay_req'])

                if p is None:
                    success = False
                    break

                paths[(u, v)] = p

                for i in range(len(p)-1):
                    G_copy[p[i]][p[i+1]]['bandwidth'] -= service[u][v]['bandwidth_req']
                    bandwidth_used_service += service[u][v]['bandwidth_req']

        # ======================
        # REWARD + COST
        # ======================
        if success:
            cpu_sum = sum(service.nodes[n]['cpu_req'] for n in service.nodes())
            storage_sum = sum(service.nodes[n]['storage_req'] for n in service.nodes())
            bw_sum = sum(service[u][v]['bandwidth_req'] for u,v in service.edges())

            revenue = cpu_sum + storage_sum + bw_sum

            cost = cpu_sum + storage_sum
            for (u,v), path in paths.items():
                hops = len(path) - 1
                cost += hops * service[u][v]['bandwidth_req']

            metrics.total_revenue += revenue
            metrics.total_cost += cost

            metrics.success += 1

            # count only nodes/links that were actually used
            active_nodes = len(mapping) if len(mapping) > 0 else 1
            active_links = len(paths) if len(paths) > 0 else 1

            avg_cpu_capacity = 90   # approx avg from generation
            avg_storage_capacity = 90
            avg_bandwidth_capacity = 90

            metrics.cpu_util_total += cpu_used_service / (active_nodes * avg_cpu_capacity + 1e-8)
            metrics.storage_util_total += storage_used_service / (active_nodes * avg_storage_capacity + 1e-8)
            metrics.bandwidth_util_total += bandwidth_used_service / (active_links * avg_bandwidth_capacity + 1e-8)

        metrics.total += 1

    # ======================
    # FINAL METRICS
    # ======================
    acceptance = (metrics.success / metrics.total) * 100
    rev_cost = metrics.total_revenue / (metrics.total_cost + 1e-8)

    cpu_util = (metrics.cpu_util_total / (metrics.success + 1e-8)) * 100
    storage_util = (metrics.storage_util_total / (metrics.success + 1e-8)) * 100 
    bandwidth_util = (metrics.bandwidth_util_total / (metrics.success + 1e-8)) * 100

    print(f"\n {name} RESULTS")
    print(f"Acceptance Ratio: {acceptance:.2f}%")
    print(f"Revenue/Cost Ratio: {rev_cost:.4f}")
    print(f"CPU Utilization: {cpu_util:.2f}%")
    print(f"Storage Utilization: {storage_util:.2f}%")
    print(f"Bandwidth Utilization: {bandwidth_util:.2f}%")

# generate test data
high_test, low_test, multi_test = generate_dataset()

# test each model
test_model(high_test, "HIGH_best.weights.h5", "HIGH")
test_model(low_test, "LOW_best.weights.h5", "LOW")
test_model(multi_test, "MULTI_best.weights.h5", "MULTI")    