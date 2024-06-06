import numpy as np
import tensorflow as tf
from dagv2 import DAG

def generate_dag(optimal_indvidual, stage_name, num_nodes):
    nodes = np.empty((0), dtype = str)

    for n in range(1, num_nodes + 1):
        nodes = np.append(nodes, f'{stage_name}_{str(n)}')

    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    edges = np.split(optimal_indvidual, np.cumsum(range(num_nodes - 1)))[1:]

    v2 = 2
    for e in edges:
        v1 = 1
        for i in e:
            if i:
                dag.add_edge(f'{stage_name}_{str(v1)}', f'{stage_name}_{str(v2)}')
            v1 += 1
        v2 += 1

    for n in nodes:
        if len(dag.predecessors(n)) == 0 and len(dag.downstream(n)) == 0:
            dag.delete_node(n)
            nodes = np.delete(nodes, np.where(nodes == n)[0][0])

    return dag, nodes

def has_same_elements(x):
    return len(set(x)) <= 1

def add_convolution(input_tensor, layer_name, kernel_height=5, kernel_width=5, num_channels=1, depth=1):
    conv_layer = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=(kernel_height, kernel_width),
        strides=(1, 1),
        padding="same",
        activation='relu',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Constant(0.01),
        name=layer_name
    )(input_tensor)

    return tf.keras.layers.BatchNormalization(name=f'{layer_name}_bn')(conv_layer)

def apply_pooling(input_tensor, pool_size=(16, 16), stride_size=(2, 2)):
    pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=stride_size, padding='same')

    return pooling_layer(input_tensor)


def generate_model(individual, stages, num_nodes, bits_indices, filters, num_labels):
    tf.keras.backend.clear_session()

    input_tensor = tf.keras.Input(shape=(28, 28, 1), name='input_tensor')

    d_node = input_tensor
    for stage_name, num_node, bpi, filter in zip(stages, num_nodes, bits_indices, filters):
        indv = individual[bpi[0]:bpi[1]]
        d_node = add_convolution(d_node, f'{stage_name}_input', depth=filter)

        if not has_same_elements(indv):
            dag, nodes = generate_dag(indv, stage_name, num_node)
            without_predessors = dag.ind_node()
            without_successors = dag.all_leaves()

            input_stage = d_node
            for wop in without_predessors:
                d_node = add_convolution(input_stage, f'{wop}_conv', depth=filter)
                dag.add_conv(wop, d_node)

            for n in nodes:
                predessors = dag.predecessors(n)

                if len(predessors) == 0: continue
                elif len(predessors) > 1:
                    sum_conv = dag.get_conv(predessors[0])
                    for prd in range(1, len(predessors)):
                        sum_conv = tf.keras.layers.Add()([sum_conv, dag.get_conv(predessors[prd])])

                    d_node = add_convolution(sum_conv, f'{n}_conv', depth=filter)
                    dag.add_conv(n, d_node)
                elif predessors:
                    d_node = add_convolution(dag.get_conv(predessors[0]), f'{n}_conv', depth=filter)
                    dag.add_conv(n, d_node)

            if len(without_successors) > 1:
                sum_conv = dag.get_conv(without_successors[0])
                for suc in range(1, len(without_successors)):
                    sum_conv = tf.keras.layers.Add()([sum_conv, dag.get_conv(without_successors[suc])])

                d_node = add_convolution(sum_conv, f'{stage_name}_output', depth=filter)
            else:
                d_node = add_convolution(dag.get_conv(without_successors[0]), f'{stage_name}_output', depth=filter)

        d_node = apply_pooling(d_node)

    d_node = tf.keras.layers.Flatten()(d_node)
    d_node = tf.keras.layers.Dense(64, activation='relu')(d_node)
    output_tensor = tf.keras.layers.Dense(num_labels, activation='softmax')(d_node)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model