import networkx as nx
import numpy as np
import tensorflow as tf
import random
import copy

# =========================
# CONFIG
# =========================
NUM_NODES = 50
EPOCHS = 100
LEARNING_RATE = 0.005

ThreNodes = 8
ThreRes = 35
ThreLat = 8

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
# 🔥 NODE FAILURE (NEW)
# =========================
def simulate_node_failure(G, prob=0.05):
    failed_nodes = []

    for node in list(G.nodes()):
        if np.random.rand() < prob:
            failed_nodes.append(node)

    for node in failed_nodes:
        if node in G:
            G.remove_node(node)

    return G

# =========================
# SERVICE GENERATION
# =========================
def generate_service():
    while True:
        n = random.randint(2, 10)
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
# CLASSIFICATION
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
# POLICY
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

# =========================
# UTILITIES
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

def can_allocate(node, demand):
    return node['cpu'] >= demand['cpu_req'] and \
           node['storage'] >= demand['storage_req'] and \
           node['delay'] <= demand['delay_req']

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

# =========================
# REWARD
# =========================
def compute_reward(service, paths):
    cpu = sum(service.nodes[n]['cpu_req'] for n in service.nodes())
    storage = sum(service.nodes[n]['storage_req'] for n in service.nodes())
    bw = sum(service[u][v]['bandwidth_req'] for u, v in service.edges())

    revenue = cpu + storage + bw
    cost = cpu + storage + sum(len(p)-1 for p in paths.values())

    return np.clip(revenue / (cost + 1e-6), -5, 5)

# =========================
# TRAIN
# =========================
def train_class(services, name):
    print(f"\n🔥 Training {name}")

    gnn = GNNEncoder()
    policy = PolicyNetwork()
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    dummy_X = tf.random.normal((NUM_NODES, 3))
    dummy_A = tf.random.normal((NUM_NODES, NUM_NODES))
    _ = gnn(dummy_X, dummy_A)
    _ = policy(tf.random.normal((NUM_NODES, 32)))

    variables = gnn.trainable_variables + policy.trainable_variables
    best_reward = -float('inf')

    for epoch in range(EPOCHS):

        G = generate_iot_graph()
        total_reward = 0
        success = 0
        baseline = 0

        accum_grads = [tf.zeros_like(v) for v in variables]
        batch_size = 100
        count = 0

        for service in services:

            G_copy = copy.deepcopy(G)

            # 🔥 NODE FAILURE APPLIED
            G_copy = simulate_node_failure(G_copy)

            if len(G_copy.nodes()) == 0:
                continue

            nodes = list(G_copy.nodes())
            if len(nodes) == 0:
                continue

            X, nodes = build_features(G_copy)
            A = build_adj(G_copy, nodes)

            with tf.GradientTape() as tape:

                emb = gnn(X, A)
                probs = policy(emb)

                mapping = {}
                chosen = []
                success_flag = True
                entropy_total = 0

                for s in service.nodes():
                    demand = service.nodes[s]

                    mask = np.array([1 if can_allocate(G_copy.nodes[n], demand) else 0 for n in nodes])
                    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

                    masked = probs * mask
                    if tf.reduce_sum(masked) <= 0:
                        success_flag = False
                        break

                    masked /= tf.reduce_sum(masked)

                    entropy_total += -tf.reduce_sum(masked * tf.math.log(masked + 1e-8))

                    idx = tf.random.categorical(tf.math.log([masked]), 1)[0][0]
                    idx = tf.clip_by_value(idx, 0, len(nodes) - 1)

                    chosen.append(tf.math.log(masked[idx] + 1e-8))

                    node = nodes[int(idx.numpy())]
                    mapping[s] = node

                    G_copy.nodes[node]['cpu'] -= demand['cpu_req']
                    G_copy.nodes[node]['storage'] -= demand['storage_req']

                paths = {}

                if success_flag:
                    for u,v in service.edges():
                        p = find_path(G_copy, mapping[u], mapping[v],
                                      service[u][v]['bandwidth_req'],
                                      service[u][v]['delay_req'])
                        if p is None:
                            success_flag = False
                            break
                        paths[(u,v)] = p
                        for i in range(len(p)-1):
                            G_copy[p[i]][p[i+1]]['bandwidth'] -= service[u][v]['bandwidth_req']

                if success_flag:
                    reward = compute_reward(service, paths)
                    success += 1
                else:
                    reward = -1

                reward = tf.convert_to_tensor(reward, dtype=tf.float32)

                if len(chosen) == 0:
                    continue

                baseline = 0.9 * baseline + 0.1 * float(reward.numpy())
                advantage = reward - baseline

                loss = -advantage * tf.reduce_sum(tf.stack(chosen)) - 0.01 * entropy_total

            grads = tape.gradient(loss, variables)

            if any(g is not None for g in grads):
                accum_grads = [acc + (g if g is not None else 0)
                               for acc, g in zip(accum_grads, grads)]

            count += 1

            if count % batch_size == 0:
                optimizer.apply_gradients(zip(accum_grads, variables))
                accum_grads = [tf.zeros_like(v) for v in variables]

            total_reward += float(reward.numpy())

        if count % batch_size != 0:
            optimizer.apply_gradients(zip(accum_grads, variables))

        acc = success / len(services)

        print(f"{name} Epoch {epoch+1} | Reward: {total_reward:.2f} | Success: {success}/{len(services)} ({acc:.2f})")

        if total_reward > best_reward:
            best_reward = total_reward
            policy.save_weights(f"{name}_best.weights.h5")
            print(f"✅ Saved best {name} model")

# =========================
# RUN
# =========================
high, low, multi = generate_dataset()

train_class(high, "HIGH")
train_class(low, "LOW")
train_class(multi, "MULTI")