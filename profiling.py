import cProfile

import pstats

from plagiarism_check import calculate_similarity

# 准备测试数据

with open('test_data.txt', 'r', encoding='utf-8') as f:

text1 = f.read()

text2 = text1[:int(len(text1)*0.8)]  # 模拟80%相似度的文本

# 性能分析

profiler = cProfile.Profile()

profiler.enable()

calculate_similarity(text1, text2)

profiler.disable()

# 输出分析结果

stats = pstats.Stats(profiler).sort_stats('cumulative')

stats.print_stats()
