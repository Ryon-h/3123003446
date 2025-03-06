import sys
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f'Error reading file {file_path}: {str(e)}')
        sys.exit(1)


def calculate_similarity(text1, text2):
    """计算两段文本的相似度"""
    # 使用结巴分词
    words1 = ' '.join(jieba.cut(text1))
    words2 = ' '.join(jieba.cut(text2))

    # 创建TF-IDF向量
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([words1, words2])

    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


def save_result(similarity, output_path):
    """保存查重结果"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f'{similarity:.2f}')
    except Exception as e:
        print(f'Error writing to file {output_path}: {str(e)}')
        sys.exit(1)


def main():
    # 检查命令行参数
    if len(sys.argv) != 4:
        print('Usage: python plagiarism_check.py <original_file> <plagiarized_file> <output_file>')
        sys.exit(1)

    # 获取文件路径
    original_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    output_path = sys.argv[3]

    # 读取文件内容
    original_text = read_file(original_path)
    plagiarized_text = read_file(plagiarized_path)

    # 计算相似度
    similarity = calculate_similarity(original_text, plagiarized_text)

    # 保存结果
    save_result(similarity, output_path)


if __name__ == '__main__':
    main()
