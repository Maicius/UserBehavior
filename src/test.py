import re
import unittest
from src.util import extract_age_pattern
from src.main import UserBehavior

class UBTest(unittest.TestCase):
    def test_extract_age(self):
        test_str = ['[1,17]', '>=60', '[25,34]']
        for item in test_str:
            res = re.findall(extract_age_pattern, item)[0]
            res = list(filter(lambda x: x != '', res))
            res = int(sum(map(int, res)) / len(res))
            print(res)

    def test_age_to_vector(self):
        test_str = ['[1,17]', '>=60', '[25,34]']
        for item in test_str:
            print(UserBehavior.age_to_vector(item))

    def test_carrer_to_vector(self):
        test_float = [1.0, 3.0, 7.0, 10.0]
        for item in test_float:
            print(UserBehavior.career_to_vector(item))

    def test_stage_to_vector(self):
        test_str = ['1,2,3', '5', '2, 4, 6, 10']
        for item in test_str:
            print(UserBehavior.stage_to_vector(item))

if __name__ =='__main__':
    unittest.main()