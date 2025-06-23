from typing import List, Optional
from collections import Counter
import re
from corl.open_r1.rewards.r_utils import extract_answer_letter_from_response


class MCQAnswerExtractor:
    def __init__(self, valid_options: List[str] = None):
        self.valid_options = valid_options or ['A', 'B', 'C', 'D', 'E', 'F']
        self.patterns = {
            'answer_tags': re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
            'think_tags': re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
            'answer_prefix': re.compile(r'.*?\b(?:Answer|Ans):\s*([A-F])', re.DOTALL | re.IGNORECASE),
            'letter_pattern': re.compile(r'\b([A-F])\b')
        }

    def extract_answer(self, content: str) -> Optional[str]:
        if content == "":
            return None

        # 1. 首先尝试提取<answer>标签中的内容
        answer_match = self.patterns['answer_tags'].search(content)
        if answer_match:
            answer = answer_match.group(1).strip()
            if len(answer) == 1:
                return self._validate_option(answer)
            else:
                return self._extract_letter_from_text(answer)

        # 2. 查找"Answer:"后的内容
        answer_prefix_match = self.patterns['answer_prefix'].search(content)
        if answer_prefix_match:
            return self._validate_option(answer_prefix_match.group(1).strip())

        answer = extract_answer_letter_from_response(content)
        if len(answer) == 1:
            return answer
        else:
            return self._extract_letter_from_text(answer)

    def _extract_letter_from_text(self, text: str) -> Optional[str]:
        """从文本中提取有效的选项字母"""
        if not text:
            return None

        # 查找所有独立选项的大写字母
        letters = self.patterns['letter_pattern'].findall(text)
        if len(letters) != 0:
            return Counter(letters).most_common(1)[0][0]

        return None

    def _validate_option(self, option: str) -> Optional[str]:
        """验证选项是否有效"""
        return option if option in self.valid_options else None
