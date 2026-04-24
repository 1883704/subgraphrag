import os
import json
from collections import defaultdict

import torch

from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.dataset.treescorer import TreeScorerDataset
from src.model.retriever import Retriever
from src.model.TreeScorer import TreeScorer
from src.setup import set_seed, prepare_sample


def _latest_checkpoint_prefix(dataset, model_type):
    if dataset is None:
        return None
    if model_type == 'treescorer':
        return f'{dataset}_tree'
    if model_type == 'retriever':
        return f'{dataset}_'
    return dataset


def resolve_checkpoint_path(args):
    if args.path and args.path != 'latest' and not args.latest:
        return args.path

    prefix = args.latest_prefix or _latest_checkpoint_prefix(
        args.dataset, args.latest_type)
    candidates = []
    for name in os.listdir('.'):
        if not os.path.isdir(name):
            continue
        if prefix and not name.startswith(prefix):
            continue
        if (
            args.latest_type == 'retriever'
            and args.dataset is not None
            and name.startswith(f'{args.dataset}_tree')
        ):
            continue
        cpt_path = os.path.join(name, 'cpt.pth')
        if os.path.exists(cpt_path):
            candidates.append(cpt_path)

    if not candidates:
        prefix_msg = f" with prefix '{prefix}'" if prefix else ''
        raise FileNotFoundError(
            f'No checkpoint directory{prefix_msg} found in {os.getcwd()}.'
        )

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    latest_path = candidates[0]
    print(f'Using latest checkpoint: {latest_path}')
    return latest_path


def _entity_and_relation_lists(raw_sample):
    entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
    relation_list = raw_sample['relation_list']
    return entity_list, relation_list


def _target_relevant_triples(raw_sample, entity_list, relation_list):
    target_relevant_triples = []
    target_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
    for triple_id in target_triple_ids:
        target_relevant_triples.append((
            entity_list[raw_sample['h_id_list'][triple_id]],
            relation_list[raw_sample['r_id_list'][triple_id]],
            entity_list[raw_sample['t_id_list'][triple_id]],
        ))
    return target_relevant_triples


def _format_sample_result(raw_sample, scored_triples):
    entity_list, relation_list = _entity_and_relation_lists(raw_sample)
    return {
        'question': raw_sample['question'],
        'scored_triples': scored_triples,
        'q_entity': raw_sample['q_entity'],
        'q_entity_in_graph': [
            entity_list[e_id] for e_id in raw_sample['q_entity_id_list']
        ],
        'a_entity': raw_sample['a_entity'],
        'a_entity_in_graph': [
            entity_list[e_id] for e_id in raw_sample['a_entity_id_list']
        ],
        'max_path_length': raw_sample['max_path_length'],
        'target_relevant_triples': _target_relevant_triples(
            raw_sample, entity_list, relation_list
        ),
    }


@torch.no_grad()
def run_retriever_inference(args, cpt, device):
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list, relation_list = _entity_and_relation_lists(raw_sample)
        top_K_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(args.max_K, len(pred_triple_scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))

        pred_dict[raw_sample['id']] = _format_sample_result(raw_sample, top_K_triples)

    root_path = os.path.dirname(args.path)
    torch.save(pred_dict, os.path.join(root_path, 'retrieval_result.pth'))


def _emb_dict_from_retriever_set(infer_set):
    emb_dict = {}
    for sample in infer_set.processed_dict_list:
        emb_dict[sample['id']] = {
            'q_emb': sample['q_emb'],
            'entity_embs': sample['entity_embs'],
            'relation_embs': sample['relation_embs'],
        }
    return emb_dict


def _triple_from_id(raw_sample, triple_id):
    entity_list, relation_list = _entity_and_relation_lists(raw_sample)
    return (
        entity_list[raw_sample['h_id_list'][triple_id]],
        relation_list[raw_sample['r_id_list'][triple_id]],
        entity_list[raw_sample['t_id_list'][triple_id]],
    )


def _score_path_triples(score_dict, raw_sample, path_node_ids, path_rel_ids, score, path_triple_ids=None):
    for edge_idx, rel_id in enumerate(path_rel_ids):
        if path_triple_ids is None:
            entity_list, relation_list = _entity_and_relation_lists(raw_sample)
            triple = (
                entity_list[path_node_ids[edge_idx]],
                relation_list[rel_id],
                entity_list[path_node_ids[edge_idx + 1]],
            )
        else:
            triple = _triple_from_id(raw_sample, path_triple_ids[edge_idx])
        score_dict[triple] = max(score_dict.get(triple, 0.0), score)


def _path_entry(raw_sample, path_node_ids, path_rel_ids, score, path_triple_ids=None):
    entity_list, relation_list = _entity_and_relation_lists(raw_sample)
    nodes = [entity_list[node_id] for node_id in path_node_ids]
    triples = []
    for edge_idx, rel_id in enumerate(path_rel_ids):
        if path_triple_ids is None:
            triples.append([
                entity_list[path_node_ids[edge_idx]],
                relation_list[rel_id],
                entity_list[path_node_ids[edge_idx + 1]],
            ])
        else:
            triples.append(list(_triple_from_id(raw_sample, path_triple_ids[edge_idx])))

    return {
        'score': score,
        'root': nodes[0] if nodes else '',
        'leaf': nodes[-1] if nodes else '',
        'nodes': nodes,
        'triples': triples,
    }


def _triple_key(triple):
    return tuple(triple)


def _path_key(path):
    return tuple(_triple_key(triple) for triple in path['triples'])


def _path_overlaps(seed_path, candidate_path):
    if seed_path['root'] != candidate_path['root']:
        return False

    seed_nodes = seed_path['nodes']
    candidate_nodes = candidate_path['nodes']
    min_len = min(len(seed_nodes), len(candidate_nodes))
    if min_len > 1 and seed_nodes[:min_len] == candidate_nodes[:min_len]:
        return True

    return len(set(seed_nodes) & set(candidate_nodes)) > 1


def _tree_score(paths):
    scores = [path['score'] for path in paths]
    return 0.7 * max(scores) + 0.3 * (sum(scores) / len(scores))


def _unique_triples(paths, max_triples):
    triples = []
    seen = set()
    for path in paths:
        for triple in path['triples']:
            key = _triple_key(triple)
            if key in seen:
                continue
            seen.add(key)
            triples.append(triple)
            if len(triples) >= max_triples:
                return triples
    return triples


def _merge_paths_to_trees(path_entries, num_trees, max_paths_per_tree, max_triples_per_tree):
    sorted_paths = sorted(path_entries, key=lambda path: path['score'], reverse=True)
    trees = []
    used_seed_keys = set()

    for seed_path in sorted_paths:
        seed_key = _path_key(seed_path)
        if seed_key in used_seed_keys:
            continue

        tree_paths = [seed_path]
        used_in_tree = {seed_key}
        for candidate_path in sorted_paths:
            candidate_key = _path_key(candidate_path)
            if candidate_key in used_in_tree:
                continue
            if not _path_overlaps(seed_path, candidate_path):
                continue

            candidate_tree_paths = tree_paths + [candidate_path]
            if len(candidate_tree_paths) > max_paths_per_tree:
                break
            if len(_unique_triples(candidate_tree_paths, max_triples_per_tree + 1)) > max_triples_per_tree:
                continue

            tree_paths.append(candidate_path)
            used_in_tree.add(candidate_key)

        tree = {
            'tree_id': len(trees),
            'score': _tree_score(tree_paths),
            'root': seed_path['root'],
            'paths': tree_paths,
            'triples': _unique_triples(tree_paths, max_triples_per_tree),
        }
        trees.append(tree)
        used_seed_keys.add(seed_key)

        if len(trees) >= num_trees:
            break

    return trees


def _write_tree_jsonl(tree_pred_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample_id, sample in tree_pred_dict.items():
            row = {'id': sample_id}
            row.update(sample)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


@torch.no_grad()
def run_treescorer_inference(args, cpt, device):
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    raw_samples = infer_set.processed_dict_list
    emb_dict = _emb_dict_from_retriever_set(infer_set)

    tree_config = config['treescorer']
    cache_dir = os.path.join(
        'data_files', config['dataset']['name'], 'cache', 'treescorer')
    cache_path = os.path.join(
        cache_dir,
        f"trees_{config['dataset']['name']}_test_h{tree_config['max_hops']}_test.pt",
    )
    tree_set = TreeScorerDataset(
        raw_samples,
        emb_dict,
        max_hops=tree_config['max_hops'],
        max_paths_per_root=tree_config['max_paths_per_root'],
        max_paths_per_sample=tree_config['max_paths_per_sample'],
        add_reverse_edges=tree_config['add_reverse_edges'],
        mode='test',
        max_neg_per_pos=tree_config['max_neg_per_pos'],
        max_neg_per_sample=tree_config['max_neg_per_sample'],
        cache_path=cache_path,
        use_cache=tree_config['use_cache'],
        cache_version=tree_config['cache_version'],
    )

    emb_size = raw_samples[0]['q_emb'].shape[-1]
    model = TreeScorer(
        emb_size=emb_size,
        hidden_size=tree_config['hidden_size'],
        num_layers=tree_config['num_layers'],
        heads=tree_config['heads'],
    ).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model.eval()

    triple_score_dicts = [defaultdict(float) for _ in raw_samples]
    path_score_lists = [[] for _ in raw_samples]
    for tree_data in tqdm(tree_set):
        sample_idx = tree_data.sample_idx.item()
        tree_data = tree_data.to(device)
        score = torch.sigmoid(model(tree_data).reshape(-1))[0].item()

        path_node_ids = tree_data.path_node_ids.detach().cpu().tolist()
        path_rel_ids = tree_data.path_rel_ids.detach().cpu().tolist()
        path_triple_ids = getattr(tree_data, 'path_triple_ids', None)
        if path_triple_ids is not None:
            path_triple_ids = path_triple_ids.detach().cpu().tolist()
        _score_path_triples(
            triple_score_dicts[sample_idx],
            raw_samples[sample_idx],
            path_node_ids,
            path_rel_ids,
            score,
            path_triple_ids,
        )
        path_score_lists[sample_idx].append(
            _path_entry(
                raw_samples[sample_idx],
                path_node_ids,
                path_rel_ids,
                score,
                path_triple_ids,
            )
        )

    pred_dict = {}
    tree_pred_dict = {}
    for i, raw_sample in enumerate(raw_samples):
        top_K_triples = [
            (triple[0], triple[1], triple[2], score)
            for triple, score in sorted(
                triple_score_dicts[i].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:args.max_K]
        ]
        pred_dict[raw_sample['id']] = _format_sample_result(raw_sample, top_K_triples)

        tree_pred_dict[raw_sample['id']] = {
            'question': raw_sample['question'],
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': pred_dict[raw_sample['id']]['q_entity_in_graph'],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': pred_dict[raw_sample['id']]['a_entity_in_graph'],
            'max_path_length': raw_sample['max_path_length'],
            'scored_trees': _merge_paths_to_trees(
                path_score_lists[i],
                args.num_trees,
                args.max_paths_per_tree,
                args.max_triples_per_tree,
            ),
        }

    root_path = os.path.dirname(args.path)
    torch.save(pred_dict, os.path.join(root_path, 'retrieval_result.pth'))
    torch.save(tree_pred_dict, os.path.join(root_path, 'tree_retrieval_result.pth'))
    _write_tree_jsonl(
        tree_pred_dict,
        os.path.join(root_path, 'tree_retrieval_result.jsonl'),
    )


@torch.no_grad()
def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.path = resolve_checkpoint_path(args)
    cpt = torch.load(args.path, map_location='cpu')

    if not isinstance(cpt, dict) or 'config' not in cpt:
        raise ValueError(
            'Checkpoint must be a dict with a config. '
            'Use train.py or train_tree.py to create cpt.pth.'
        )

    config = cpt['config']
    if cpt.get('model_type') == 'treescorer' or 'treescorer' in config:
        run_treescorer_inference(args, cpt, device)
    else:
        run_retriever_inference(args, cpt, device)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth. Use "latest" with -d to select the newest checkpoint.')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        help='Dataset name used with --latest or -p latest')
    parser.add_argument('--latest', action='store_true',
                        help='Use the newest checkpoint directory in the current folder')
    parser.add_argument('--latest_type', type=str, default='treescorer',
                        choices=['treescorer', 'retriever', 'any'],
                        help='Checkpoint prefix type used by --latest')
    parser.add_argument('--latest_prefix', type=str, default=None,
                        help='Custom checkpoint directory prefix used by --latest')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    parser.add_argument('--num_trees', type=int, default=5,
                        help='Number of reasoning trees to keep for TreeScorer inference')
    parser.add_argument('--max_paths_per_tree', type=int, default=5,
                        help='Maximum number of paths merged into each reasoning tree')
    parser.add_argument('--max_triples_per_tree', type=int, default=30,
                        help='Maximum number of unique triples in each reasoning tree')
    args = parser.parse_args()

    if not args.path and not args.latest:
        parser.error('one of -p/--path or --latest is required')
    
    main(args)
