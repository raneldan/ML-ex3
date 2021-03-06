from typing import List
import re


class RowToVector:
    def __init__(self, row: List[str], is_email: bool = False):
        self.row: List[str] = [word for line in row for word in line.split()]
        self.funcs: List = []
        self.vector = []
        self.__init_funcs(is_email)
        for func in self.funcs:
            self.vector.append(func())

    def __init_funcs(self, is_email: bool = False):
        if is_email:
            # self.funcs.append(self.number_of_long_sentences)
            self.funcs.append(self.number_of_money_ref)
            self.funcs.append(self.number_of_marketing_ref)
        else:
            self.funcs.append(self.has_free)
            self.funcs.append(self.has_price)

        #self.funcs.append(self.number_of_words)
        self.funcs.append(self.percentage_of_non_letters)
        self.funcs.append(self.has_link)
        self.funcs.append(self.percentage_of_capital_letters)
        #self.funcs.append(self.num_of_numbers)
        self.funcs.append(self.has_re)
        self.funcs.append(self.has_no_title)
        self.funcs.append(self.has_xxx)
        self.funcs.append(self.has_txt)

    def number_of_words(self) -> int:
        return len(self.row)

    def percentage_of_non_letters(self) -> float:
        non_letter_sign = 0
        for word in self.row:
            for index, letter in enumerate(word):
                if ord(word[index]) > 122 or ord(word[index]) < 65:
                    non_letter_sign += 1
        p =  non_letter_sign / self.num_of_characters()
        ans = 1 if p>50 else 0
        return ans

    def number_of_long_sentences(self):
        sentences = ' '.join(self.row).split('.')
        return sum((list(map(lambda x: 1 if (len(x.split(' ')) > 4) else 0, sentences))))

    def has_link(self) -> int:
        return 1 if len(re.findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            ''.join(self.row))) > 0 else 0

    def percentage_of_capital_letters(self) -> float:
        capital_letters = 0
        for word in self.row:
            for index, letter in enumerate(word):
                if ord(word[index]) > 64 and ord(word[index]) < 91:
                    capital_letters += 1
        p = capital_letters / self.num_of_characters()
        ans = 1 if p>50 else 0
        return ans

    def num_of_characters(self):
        number_of_characters = 0
        for word in self.row:
            for letter in word:
                number_of_characters += len(letter)
        return number_of_characters

    def num_of_numbers(self):
        numbers = 0
        for word in self.row:
            for index, letter in enumerate(word):
                if ord(word[index]) >= 48 and ord(word[index]) <= 57:
                    numbers += 1
        ans = 1 if numbers > 10 else 0
        return numbers

    def number_of_marketing_ref(self):
        marketing_items = (
            re.findall('click|email|sales|below|here|subscribe|Visit|our|website|promotion|Member', ''.join(self.row),
                       re.IGNORECASE))
        return 0 if ((not marketing_items) or marketing_items[0] == '') else 1

    def number_of_money_ref(self):
        money_items = (
            re.findall('free|money|$|price|offer|special|now|award|deal|cheap|save', ''.join(self.row), re.IGNORECASE))
        return 1 if money_items[0] != '' else 0

    def has_free(self):
        return 1 if re.search('free', ''.join(self.row)) else 0

    def has_xxx(self):
        return 1 if re.search('xxx', ''.join(self.row), re.IGNORECASE) else 0

    def has_price(self):
        return 1 if re.search('$', ''.join(self.row)) else 0

    def has_txt(self):
        return 1 if re.search('txt', ''.join(self.row)) else 0

    def has_re(self):
        return 1 if re.search('Subject:re:', ''.join(self.row)) else 0

    def has_no_title(self):
        return 0 if re.search('Subject', ''.join(self.row)) else 1

    def format(self):
        return (self.vector)


r2v = RowToVector(["assd ffff ggg gggg ggg ggg gggg. sd."], True)
print(r2v.number_of_long_sentences())
