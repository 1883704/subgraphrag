#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理ChatDoctor数据集的脚本
为每个样本构建包含BFS k跳内所有实体的图
"""

import json
import collections
from typing import Dict, List, Set, Tuple, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatDoctorProcessor:
    def __init__(self, kg_file: str, entity_file: str, relation_file: str):
        """
        初始化处理器

        Args:
            kg_file: 知识图谱三元组文件路径
            entity_file: 实体到ID映射文件路径
            relation_file: 关系到ID映射文件路径
        """
        self.kg_file = kg_file
        self.entity_file = entity_file
        self.relation_file = relation_file

        # 加载知识图谱
        self.kg_graph = self._load_knowledge_graph()
        self.entity2id = self._load_entity_mapping()
        self.relation2id = self._load_relation_mapping()

        # 实体匹配统计
        self.total_entities = 0
        self.matched_entities = 0
        self.unmatched_entities = set()

        logger.info(f"加载了 {len(self.kg_graph)} 个三元组")
        logger.info(f"加载了 {len(self.entity2id)} 个实体")
        logger.info(f"加载了 {len(self.relation2id)} 种关系")

    def _load_knowledge_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        加载知识图谱，构建邻接表

        Returns:
            邻接表字典，key为实体，value为(关系, 目标实体)的列表
        """
        kg_graph = collections.defaultdict(list)

        with open(self.kg_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 3:
                        head, relation, tail = parts
                        # 只添加正向边（有向图）
                        kg_graph[head].append((relation, tail))

        return dict(kg_graph)

    def _load_entity_mapping(self) -> Dict[str, int]:
        """加载实体到ID的映射"""
        entity2id = {}
        with open(self.entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        entity, entity_id = parts
                        entity2id[entity] = int(entity_id)
        return entity2id

    def _load_relation_mapping(self) -> Dict[str, int]:
        """加载关系到ID的映射"""
        relation2id = {}
        with open(self.relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        relation, relation_id = parts
                        relation2id[relation] = int(relation_id)
        return relation2id

    @staticmethod
    def _parse_entities(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, list):
            return [e.strip() for e in val if isinstance(e, str) and e.strip()]
        if isinstance(val, str):
            return [e.strip() for e in val.split(',') if e.strip()]
        return []

    @staticmethod
    def _variants(entity: str) -> List[str]:
        return [
            entity,
            entity.replace(' ', '_'),
            entity.replace('_', ' '),
            entity.lower(),
            entity.upper(),
            entity.title()
        ]

    def bfs_search(self, start_entities: List[str], max_hops: int = 5) -> Set[Tuple[str, str, str]]:
        """
        从起始实体进行BFS搜索，获取指定跳数内的所有三元组

        Args:
            start_entities: 起始实体列表
            max_hops: 最大跳数

        Returns:
            包含所有相关三元组的集合
        """
        if not start_entities:
            return set()

        visited = {}
        relevant_triples = set()
        queue = collections.deque()
        for entity in start_entities:
            if entity in self.kg_graph:
                queue.append((entity, 0))
                visited[entity] = 0

        while queue:
            current_entity, current_hop = queue.popleft()
            if current_hop >= max_hops:
                continue
            # 正向边
            if current_entity in self.kg_graph:
                for relation, neighbor in self.kg_graph[current_entity]:
                    relevant_triples.add((current_entity, relation, neighbor))
                    if neighbor not in visited and current_hop + 1 < max_hops:
                        visited[neighbor] = current_hop + 1
                        queue.append((neighbor, current_hop + 1))

        max_hop_in_result = max([visited.get(entity, 0) for entity in visited]) if visited else 0
        logger.info(
            f"从 {len(start_entities)} 个起始实体搜索到 {len(relevant_triples)} 个三元组 (max_hops={max_hops}, 实际最大跳数={max_hop_in_result})")

        return relevant_triples

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本，添加graph字段

        Args:
            sample: 包含input, output, input_KG, output_KG的样本

        Returns:
            添加了graph字段的样本，字段顺序为：question, answer, q_entity, a_entity, graph
        """
        # 提取并解析 q_entities
        q_entities_raw = self._parse_entities(sample.get('input_KG', ''))
        q_entities_matched = []
        for entity in q_entities_raw:
            self.total_entities += 1
            found = None
            for fmt in self._variants(entity):
                if fmt in self.kg_graph:
                    found = fmt
                    break
            if found:
                q_entities_matched.append(found)
                self.matched_entities += 1
            else:
                self.unmatched_entities.add(entity)

        start_entities = q_entities_matched
        if not start_entities:
            logger.warning(f"样本中没有任何实体能在知识图谱中找到匹配: {q_entities_raw}")
            graph = []
        else:
            graph_triples = self.bfs_search(start_entities)
            graph = []
            for head, relation, tail in graph_triples:
                graph.append({
                    'head': head,
                    'relation': relation,
                    'tail': tail
                })

        # 根据图中的实体对齐 a_entity，并确保 answer == a_entity
        graph_entities = set()
        for t in graph:
            graph_entities.add(t['head'])
            graph_entities.add(t['tail'])
        a_entities_raw = self._parse_entities(sample.get('output_KG', ''))
        a_entities_aligned = []
        for ent in a_entities_raw:
            matched = None
            for fmt in self._variants(ent):
                if fmt in graph_entities:
                    matched = fmt
                    break
            if matched:
                a_entities_aligned.append(matched)
        # 去重，保持顺序
        seen = set()
        a_entities_final = []
        for e in a_entities_aligned:
            if e not in seen:
                seen.add(e)
                a_entities_final.append(e)

        # 输出为字符串（逗号分隔），与原下游准备脚本兼容
        q_entity_out = ', '.join(q_entities_matched)
        a_entity_out = ', '.join(a_entities_final)
        answer_out = a_entity_out

        processed_sample = {
            'question': sample.get('input', ''),
            'answer': answer_out,
            'q_entity': q_entity_out,
            'a_entity': a_entity_out,
            'graph': graph
        }

        return processed_sample

    def process_dataset(self, input_file: str, output_file: str):
        """
        处理整个数据集

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        logger.info(f"开始处理数据集: {input_file}")

        processed_samples = []
        total_samples = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    total_samples += 1
                    processed_sample = self.process_sample(sample)
                    processed_samples.append(processed_sample)
                    if total_samples % 100 == 0:
                        logger.info(f"已处理 {total_samples} 个样本")
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行JSON解析失败: {e}")
                    continue
                except Exception as e:
                    logger.error(f"处理第 {line_num} 行时出错: {e}")
                    continue

        logger.info(f"开始保存处理后的数据集到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in processed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"数据集处理完成！总共处理了 {total_samples} 个样本")
        logger.info(f"输出文件: {output_file}")

        self._print_statistics(processed_samples)

    def _print_statistics(self, processed_samples: List[Dict[str, Any]]):
        """打印统计信息"""
        if not processed_samples:
            return

        graph_sizes = [len(sample.get('graph', [])) for sample in processed_samples]

        logger.info("=== 统计信息 ===")
        logger.info(f"总样本数: {len(processed_samples)}")
        logger.info(f"平均图大小: {sum(graph_sizes) / len(graph_sizes):.2f}")
        logger.info(f"最大图大小: {max(graph_sizes)}")
        logger.info(f"最小图大小: {min(graph_sizes)}")

        all_entities_in_graphs = set()
        for sample in processed_samples:
            graph = sample.get('graph', [])
            for triple in graph:
                all_entities_in_graphs.add(triple['head'])
                all_entities_in_graphs.add(triple['tail'])

        logger.info(f"图中包含的实体数: {len(all_entities_in_graphs)}")
        logger.info(f"知识图谱总实体数: {len(self.kg_graph)}")
        logger.info(f"实体覆盖率: {len(all_entities_in_graphs) / len(self.kg_graph) * 100:.2f}%")

        logger.info("=== 实体匹配统计 ===")
        logger.info(f"总实体数: {self.total_entities}")
        logger.info(f"成功匹配: {self.matched_entities}")
        logger.info(f"匹配失败: {len(self.unmatched_entities)}")
        if self.total_entities > 0:
            logger.info(f"实体匹配率: {self.matched_entities / self.total_entities * 100:.2f}%")

        if self.unmatched_entities:
            sample_unmatched = list(self.unmatched_entities)[:10]
            logger.info(f"未匹配实体示例: {', '.join(sample_unmatched)}")
            if len(self.unmatched_entities) > 10:
                logger.info(f"... 还有 {len(self.unmatched_entities) - 10} 个未匹配实体")


def main():
    """主函数"""
    # 文件路径
    kg_file = "../raw_data/chatdoctor5k/train.txt"
    entity_file = "../raw_data/chatdoctor5k/entity2id.txt"
    relation_file = "../raw_data/chatdoctor5k/relation2id.txt"
    input_file = "../raw_data/chatdoctor5k/newchatdoctor5k_KG.json"
    output_file = "../data/chatdoctor5k/newchatdoctor5k.json"

    try:
        processor = ChatDoctorProcessor(kg_file, entity_file, relation_file)
        processor.process_dataset(input_file, output_file)
        logger.info("数据集处理完成！")

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()