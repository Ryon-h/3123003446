import unittest
from plagiarism_check import calculate_similarity

class TestPlagiarismCheck(unittest.TestCase):
    def test_identical_texts(self):
        """测试完全相同的文本"""
        text = "这是一个测试用例"
        similarity = calculate_similarity(text, text)
        self.assertEqual(similarity, 1.0)

    def test_completely_different_texts(self):
        """测试完全不同的文本"""
        text1 = "今天天气真好"
        text2 = "学习编程很有趣"
        similarity = calculate_similarity(text1, text2)
        self.assertLess(similarity, 0.5)

    def test_partially_similar_texts(self):
        """测试部分相似的文本"""
        text1 = "我喜欢吃苹果和香蕉"
        text2 = "我喜欢吃香蕉和橙子"
        similarity = calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 1.0)

    def test_empty_texts(self):
        """测试空文本"""
        text1 = ""
        text2 = ""
        similarity = calculate_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)

    def test_special_characters(self):
        """测试包含特殊字符的文本"""
        text1 = "测试！@#￥%……&*（）"
        text2 = "测试！@#￥%……&*（）"
        similarity = calculate_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)

if __name__ == '__main__':
    unittest.main()
