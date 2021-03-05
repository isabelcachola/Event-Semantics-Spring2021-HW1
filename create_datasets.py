import argparse
import time
import csv
import os
from decomp import UDSCorpus
 
ROLES = {
    "agent": lambda p: ((p['volition']['value'] > 0) or (p['instigation']['value'] > 0 )) and \
                        (p['existed_before']['value'] > 0),
    "patient": lambda p: ((p['volition']['value'] < 0) or (p['instigation']['value'] < 0 )) and \
                        (p['existed_before']['value'] > 0),
    "theme": lambda p: ( (p['change_of_location']['value'] > 0) and \
                        (p['volition']['value'] < 0) ) and \
                        (p['existed_before']['value'] > 0) ,
    "experiencer": lambda p: (p['change_of_state_continuous']['value'] > 0) and \
                        (p['volition']['value'] < 0) and \
                        (p['awareness']['value'] > 0),
    "recipient": lambda p: (p['change_of_possession']['value'] > 0) and \
                        (p['existed_before']['value'] > 0) and \
                        (p['volition']['value'] < 0)
}

# Dataset file header
HEADER = [ "ITEM_ID", "GRAPH_ID", "SENTENCE", "PRED_HEAD_IDX", "ARG_HEAD_IDX", "LABEL"]

def parse_node_name(node):
    node_type, node_idx = node.split('-')[-2:] # ['arg', 'X'] or ['pred', 'X']
    return node_type, int(node_idx)

'''
Parses edge tuples, arg and pred aren't always in the same order in the edge
Example edge: ('ewt-train-12-semantics-pred-7', 'ewt-train-12-semantics-arg-3')
Returns pred_head_idx (int) arg_head_idx (int), 0 indexed
'''
def parse_edge_name(edge):
    pred_head_idx, arg_head_idx = -1, -1
    node_type, node_idx = parse_node_name(edge[0])
    if node_type == 'pred':
        pred_head_idx = node_idx - 1 
    elif node_type == 'arg':
        arg_head_idx = node_idx - 1
    
    node_type, node_idx = parse_node_name(edge[1])
    if node_type == 'pred':
        pred_head_idx = node_idx - 1
    elif node_type == 'arg':
        arg_head_idx = node_idx - 1

    return pred_head_idx, arg_head_idx

def process_dataset_split(data, role, criteria):
    dataset = []
    n_true = 0
    for graphid, graph in data.items():
        semantic_edges = graph.semantics_edges()
        for edge, properties in semantic_edges.items():
            if 'protoroles' in properties:
                pred_head_idx, arg_head_idx = parse_edge_name(edge)
                if pred_head_idx == -1 or arg_head_idx == -1:
                    pass
                else:
                    try:
                        label = int(criteria(properties['protoroles']))
                        if label:
                            n_true +=1
                        item_id = "_".join(str(x) for x in [graphid, pred_head_idx, arg_head_idx])
                        row = [ item_id, graphid, graph.sentence, pred_head_idx, arg_head_idx, label ]
                        dataset.append(row)
                    except Exception:
                        pass
    return dataset, n_true

def main(args):
    for split in ['train', 'dev', 'test']:
        data = UDSCorpus(split=split)
        for role, criteria in ROLES.items():
            print(f'Processing {role} {split}', end='   ')
            dataset, n_true = process_dataset_split(data, role, criteria)
            outfile = os.path.join(args.datadir, f'{role}-{split}.csv')
            with open(outfile, 'w') as fout:
                writer = csv.writer(fout, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(HEADER)
                writer.writerows(dataset)
            print(f'Found {n_true}/{len(dataset)} true examples')

        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', help='directory to save data')
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
    print(f'Time to run script: {end-start} secs')