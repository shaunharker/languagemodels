
import random

# Todo: binner could make exact-sized examples. how about a truncate option?
class Binner:
    def __init__(self, source, feeder, bin_size):
        self.feeder = feeder
        self.bin_size = bin_size
        self.lastitem = {'source': source, 'bytes': b'', 'mask': np.zeros((1, 1, 0))}
        
    def __iter__(self):
        return self

    def __next__(self):
        for subexample in self.feeder:
            example = {**self.lastitem}
            self.lastitem = subexample
            m = len(example['bytes'])
            n = len(subexample['bytes'])
            if n + m > self.bin_size and n > 0:
                return example
            self.lastitem['bytes'] = example['bytes'] + subexample['bytes']
            self.lastitem['mask'] = np.concatenate((example['mask'],subexample['mask']), axis=2)

        # If all generators are exhausted
        raise StopIteration
        

class Feeder:
    def __init__(self):
        self.generators = {}
        self.total_weight = 0

    def add_generator(self, name, generator, weight, bin_size=None):
        if bin_size is not None:
            generator = Binner(name, generator, bin_size)
        self.generators[name] = {"generator": generator, "weight": weight}
        self.total_weight += weight

    def remove_generator(self, name):
        if name in self.generators:
            self.total_weight -= self.generators[name]["weight"]
            del self.generators[name]

    def list_generators(self):
        return {name: {"weight": gen["weight"]} for name, gen in self.generators.items()}

    def set_weight(self, name, weight):
        if name in self.generators:
            self.total_weight += weight - self.generators[name]["weight"]
            self.generators[name]["weight"] = weight

    def __iter__(self):
        return self

    def __next__(self):
        while self.generators:
            # Choose a generator based on the provided weights
            names, weights = zip(*[(name, gen["weight"]) for name, gen in self.generators.items()])
            chosen_name = random.choices(names, weights, k=1)[0]
            chosen_generator = self.generators[chosen_name]["generator"]

            try:
                return next(chosen_generator)
            except StopIteration:
                # Remove the generator and its weight if it's exhausted
                self.remove_generator(chosen_name)
                # Continue the loop to try with the next generator
        # If all generators are exhausted
        raise StopIteration

def random_snippet_feeder(path, source=None, example_length=2048):
    reader = Reader(path)
    if source is None:
        source = path
    while True:
        idx = random.randint(0, len(reader.cat)-example_length)
        example = {'bytes': reader.cat[idx:idx+example_length]}
        example['source'] = source
        if b'\x00' not in example['bytes']:
            yield example


def instruction_example(prefix, suffix):
    example = prefix + suffix + bytes([0])
    mask = np.ones((1, 1, len(example)))
    mask[0,0,:len(prefix)-1] = 0
    return mask, example

import json
def make_wikipedia_feeder(path, source=None, example_length=2048):
    reader = Reader(path)
    reader.func = lambda x : json.loads(x)
    if source is None:
        source = path
    while True:
        idx = random.randint(0, len(reader)-1)
        line = reader[idx]
        title = line['meta']['title']
        # filter
        if line['meta']['language'] != 'en':
            continue
        # build training item; instruction style
        article = line['text']
        opening = article.split('\n')[0] + '\n'
        prefix = bytes(f"Input: Write a Wikipedia article about '{title}'.\nOutput: ", encoding='utf8')
        suffix = bytes(opening, encoding='utf8')[:example_length-len(prefix)]
        mask, example = instruction_example(prefix, suffix)
        training_item = {'bytes': example,
                         'source': source,
                         'mask': mask}
        yield training_item


books_path = '/samsung/data/redpajama/book/book.jsonl'
books_feeder = random_snippet_feeder(books_path, source='books', example_length=config['example_length'])
pile_path  = '/samsung/data/thepile/03.jsonl'
pile_feeder = random_snippet_feeder(pile_path, source='pile', example_length=config['example_length'])
wikipedia_path  = '/samsung/data/redpajama/wikipedia/wiki.jsonl'
wikipedia_feeder = make_wikipedia_feeder(wikipedia_path, source='wikipedia', example_length=config['example_length'])

def get_txt_examples(path):
    lines = []
    with open(path) as infile:
        lines = infile.readlines()
    return [bytes(line, encoding='utf8') for line in lines]

def cleanword(word):
    for _ in range(8):
        for c in ['_', '(', ')', '[', ']', ',', '.', ';', '?', '!', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            c = bytes(c, encoding='utf8')
            if word.startswith(c):
                word = word[1:]
            if word.endswith(c):
                word = word[:-1]
    return word

def make_dictionary_examples(path='/samsung/data/webster/dictionarylines.txt'):
    examples = get_txt_examples(path)
    result = []
    for example_idx, example in enumerate(examples):
        idx = example.find(b'/')
        word = example[:idx-1]
        word = cleanword(word)
        if word == b'':
            continue
        if b'_' in word:
            continue
        prefix = bytes(f'Input: Define "{word}".\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/dictionary",
            "bytes": training_bytes,
            "mask": mask
        }
        result.append(training_item)
    for idx in range(len(result)):
        result[idx]['idx'] = idx
    return result

def repeated_shuffle_feeder(examples):
    while True:
        random.shuffle(examples)
        for example in examples:
            yield example
            
dictionary_path = '/samsung/data/webster/dictionarylines.txt'
dictionary_feeder = repeated_shuffle_feeder(make_dictionary_examples(dictionary_path))

from itertools import zip_longest

def integercompareadvice(x, y):
    advice = f"We compare x := {x} to y := {y}.\n"
    advice += f"Observe that len(x) == {len(str(x))}.\n"
    advice += f"Observe that len(y) == {len(str(y))}.\n"
    # Note... we could technically recurse here.
    if len(str(x)) < len(str(y)):
        advice += f"We have that len(x) < len(y).\n"
        return advice
    if len(str(x)) > len(str(y)):
        advice += f"We have that len(x) > len(y).\n"
        return advice
    if len(str(x)) == len(str(y)):
        advice += f"We have that len(x) == len(y), necessitating a digit by digit check.\n"
        for (a, b) in zip(str(x), str(y)):
            if int(a) < int(b):
                advice += f"{a} < {b}.\n"
                return advice
            elif int(a) > int(b):
                advice += f"{a} > {b}.\n"
                return advice
            else:
                advice += f"{a} == {b}.\n"
        advice += "All digits identical.\n"
        return advice
    return advice

def integercompareexample(x, y):
    if x > 99 or y > 99:
        example = f"{integercompareadvice(x,y)}"
    if x == y:
        example += f"We have that {x} == {y}.\n"
    elif x < y:
        example += f"We have that {x} < {y}.\n"
    else:
        example += f"We have that {x} > {y}.\n"
    return example
     

def integercompare_feeder():
    global coremagnitudes
    while True:
        M = random.choice(coremagnitudes)
        N = random.choice(coremagnitudes)
        x = random.randint(1, M)
        y = random.randint(1, N)
        example = bytes(integercompareexample(x, y), encoding='utf8')
        prefix = bytes(f'Input: Compare the integers {x} and {y}.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/integercompare",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

def halfintegerexample(x):
    example = f"The quotient is {x//2} and the remainder is {x%2}.\n"
    return example
     
coremagnitudes_by_problem = {}
coremagnitudes_by_problem["core/halfinteger"] = [10, 100, 1000, 10000]
coremagnitudes_by_problem["core/blackboardaddition"] = [10, 100, 1000, 10000, 100000]

def halfinteger_feeder():
    while True:
        M = random.choice(coremagnitudes_by_problem["core/halfinteger"])
        x = random.randint(1, M)
        example = bytes(halfintegerexample(x), encoding='utf8')
        prefix = bytes(f'Input: What is {x} divided by 2? Give the quotient and remainder.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/halfinteger",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

from itertools import zip_longest

def blackboardadditionadvice(x, y):
    if x < y:
        x, y = y, x
    
    n = len(str(x))
    
    sy = ' '*(n-len(str(y))) + str(y)
    
    carryline = " "*(n+1) + "\n"
    bbproblem = f"  {x}\n+ {sy}\n" + "-"*(n+2) + "\n"
    resultline = " "*(n+2) + "\n"
    advice = f"\nStep 0:\n{carryline}{bbproblem}{resultline}"
    c = 0
    for k in range(n):
        a = int(str(x)[-k-1])
        try:
            b = int(str(y)[-k-1])
        except:
            b = 0
        d = (a + b + c)%10
        oldc = c
        c = (a + b + c)//10
        if c != 0:
            if k < n-1:
                carryline = carryline[:-k-2] + str(c) + carryline[-k-1:]
            else:
                resultline = ' 1' + resultline[2:]
        resultline = resultline[:-k-2] + str(d) + resultline[-k-1:]
        equat = ""
        if oldc > 0:
            equat += f"{oldc}+"
        equat += f"{a}"
        equat += f"+{b}"
        equat += f"={oldc+a+b}"
        advice += f"\nStep {k+1}: {equat}\n{carryline}{bbproblem}{resultline}"
    return advice

def blackboardadditionexample(x, y):
    example = f"{blackboardadditionadvice(x,y)}\nThe sum is {x+y}.\n"
    return example
     

def blackboardaddition_feeder():
    global coremagnitudes
    while True:
        M = random.choice(coremagnitudes_by_problem["core/blackboardaddition"])
        N = random.choice(coremagnitudes_by_problem["core/blackboardaddition"])
        x = random.randint(1, M)
        y = random.randint(1, N)
        example = bytes(blackboardadditionexample(x, y), encoding='utf8')
        prefix = bytes(f'Input: Give a blackboard demonstration of how to perform the sum {x} + {y}.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/blackboardaddition",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

from itertools import zip_longest

# def additionadvice(x, y):
#     S = 0
#     xs = str(x)
#     ys = str(y)
#     rxs = xs[::-1]
#     rys = ys[::-1]
#     rxys = str(x+y)[::-1]
    
#     L = max(len(rxs), len(rys))
    
#     pairs =  [(a,b) for (a,b) in zip_longest(rxs, rys, fillvalue='0')]
#     advice = ""
#     cc = '0'
#     for (ca,cb) in pairs:
#         a = int(ca)
#         b = int(cb)
#         c = int(cc)        
#         advice += f"{ca}+{cb}" if c == 0 else f"{cc}+{ca}+{cb}"
        
#         d = (c+a+b)%10
#         c = (c+a+b)//10

#         cc = str(c)
#         cd = str(d)
#         advice += f"={cd}," if c == 0 else f"={cc}{cd},"
#     advice = advice[:-1]
#     return advice

def additionexample(x, y):
    #example = f"«{x}+{y}=«{additionadvice(x,y)}»{x+y}»The sum is {x+y}.\n"
    example = f"The sum is {x+y}.\n"
    return example
     

def addition_feeder():
    magnitudes = [10, 100, 1000, 10000, 100000]
    while True:
        M = random.choice(magnitudes)
        N = random.choice(magnitudes)
        x = random.randint(1, M)
        y = random.randint(1, N)
        example = bytes(additionexample(x, y), encoding='utf8')
        prefix = bytes(f'Input: Compute the sum {x} + {y}.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/addition",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

def littleendianexample(x):
    #example = f"«{x}+{y}=«{additionadvice(x,y)}»{x+y}»The sum is {x+y}.\n"
    xs = str(x)
    example = f"The little-endian representation of {xs} is L{xs[::-1]}.\n"
    return example
     

def littleendian_feeder():
    magnitudes = [10, 100, 1000, 10000, 100000]
    while True:
        M = random.choice(magnitudes)
        x = random.randint(1, M)
        example = bytes(littleendianexample(x), encoding='utf8')
        optional = "(i.e. digit-reversed) " if random.randint(1,2)==1 else ""
        prefix = bytes(f'Input: What is the little-endian {optional}representation of {x}?\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/littleendian",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

def reverseadditionexample(x, y):
    #example = f"«{x}+{y}=«{additionadvice(x,y)}»{x+y}»The sum is {x+y}.\n"
    xs = str(x)
    ys = str(y)
    zs = str(x+y)
    example = f"L{xs[::-1]}+L{ys[::-1]}=L{zs[::-1]}.\nThe sum is {x+y}.\n"
    return example
     

def reverseaddition_feeder():
    magnitudes = [10, 100, 1000, 10000, 100000]
    while True:
        M = random.choice(magnitudes)
        N = random.choice(magnitudes)
        x = random.randint(1, M)
        y = random.randint(1, N)
        example = bytes(reverseadditionexample(x, y), encoding='utf8')
        prefix = bytes(f'Input: Compute the sum {x} + {y} using little-endian scratchwork.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/reverseaddition",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

def singlemultiplication_feeder():
    global coremagnitudes
    while True:
        N = random.choice(coremagnitudes)
        x = random.randint(0, 9)
        y = random.randint(0, N)
        example = bytes(f"{x*y}", encoding='utf8')
        prefix = bytes(f'Input: Compute the product {x} * {y}.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/singlemultiplication",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

def multiplicationadvice(x, y):
    S = 0
    xs = str(x)
    n = len(xs)
    advice = f""
    for idx, xsi in enumerate(xs):
        zi = int(xsi)*y
        advice += f'{xsi}*{y}={str(zi)},'
        if idx > 0:
            advice += (f'«{S}0+{zi}=«{additionadvice(S*10, zi)}»{str(S*10+zi)}»,')
        S = 10*S + zi
    return advice[:-1]
    

def multiplicationexample(x, y):
    example = f"«{x}*{y}=«{multiplicationadvice(x,y)}»{x*y}»The product is {x*y}.\n"
    return example

def multiplication_feeder():
    magnitudes = [10, 100, 1000, 10000, 100000]
    while True:
        M = random.choice(magnitudes)
        N = random.choice(magnitudes)
        x = random.randint(10, M)
        y = random.randint(1, N)
        example = bytes(multiplicationexample(x, y), encoding='utf8')
        prefix = bytes(f'Input: Compute the product {x} * {y}.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/multiplication",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

def peasantmultiplicationadvice(x, y):
    z = 0
    advice = f"{x} * {y} + {z} =\n"
    while y > 0:
        if z > 0:
            advice += f"{x} * {y} + {z} =\n"
        if y % 2 == 1:
            y = y - 1
            z = z + x
            advice += f"{x} * {y} + {z} =\n"
        y = y // 2
        x = 2 * x
    advice += f"{z}.\n"
    return advice
    

def peasantmultiplicationexample(x, y):
    example = f"{peasantmultiplicationadvice(x,y)}\nThe product is {x*y}.\n"
    return example

def peasantmultiplication_feeder():
    magnitudes = [10, 100, 300]
    while True:
        M = random.choice(magnitudes)
        N = random.choice(magnitudes)
        x = random.randint(10, M)
        y = random.randint(1, N)
        example = bytes(peasantmultiplicationexample(x, y), encoding='utf8')
        prefix = bytes(f'Input: Compute the product {x} * {y} via peasant multiplication.\nOutput: ', encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/peasantmultiplication",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item

##

import string
import sys
def generate_random_unicode_string(length=100):
    # Define the valid ASCII characters: letters, digits, punctuation (excluding single quote)
    ascii_chars = string.ascii_letters + string.digits + string.punctuation

    # Define the range for printable unicode characters (beyond ASCII)
    unicode_range = (0x0021, 0x007E)  # ASCII range
    extended_unicode = ''.join(
        chr(i) for i in range(sys.maxunicode)
        if (i > unicode_range[1]) and (chr(i).isprintable())
    )

    # Combine ASCII and extended unicode characters
    valid_chars = ascii_chars # + extended_unicode

    # Generate a random string
    random_string = ''.join(random.choice(valid_chars) for _ in range(length))
    return random_string

def generate_random_ascii_string(length=100):
    # Define the valid ASCII characters: letters, digits, punctuation (excluding single quote)
    ascii_chars = string.ascii_letters + string.digits + string.punctuation

    # Combine ASCII and extended unicode characters
    valid_chars = ascii_chars # + extended_unicode

    # Generate a random string
    random_string = ''.join(random.choice(valid_chars) for _ in range(length))
    return random_string

def generate_random_alphanumeric_string(length=100):
    # Define the valid ASCII characters: letters, digits, punctuation (excluding single quote)
    ascii_chars = string.ascii_letters + string.digits

    # Combine ASCII and extended unicode characters
    valid_chars = ascii_chars # + extended_unicode

    # Generate a random string
    random_string = ''.join(random.choice(valid_chars) for _ in range(length))
    return random_string

def generate_random_alphabet_string(length=100):
    # Define the valid ASCII characters: letters, digits, punctuation (excluding single quote)
    ascii_chars = string.ascii_letters

    # Combine ASCII and extended unicode characters
    valid_chars = ascii_chars # + extended_unicode

    # Generate a random string
    random_string = ''.join(random.choice(valid_chars) for _ in range(length))
    return random_string

def generate_random_lowercase_string(length=100):
    return generate_random_alphabet_string(length).lower()

##

string_len_level = 8

def string_reverse_feeder():
    while True:
        n = random.randint(1, string_len_level)
        s = generate_random_alphabet_string(n)
        t = s[::-1]
        example = bytes(f"'{t}'\n", encoding='utf8')
        prefix = bytes(f"Input: Reverse the string '{s}'.\nOutput: ", encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/string/reverse",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item
                       
def string_lower_feeder():
    while True:
        n = random.randint(1, string_len_level)
        s = generate_random_alphabet_string(n)
        t = s.lower()
        example = bytes(f"'{t}'\n", encoding='utf8')
        prefix = bytes(f"Input: Lowercase the string '{s}'.\nOutput: ", encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/string/lower",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item
                       
def string_upper_feeder():
    while True:
        n = random.randint(1, string_len_level)
        s = generate_random_alphabet_string(n)
        t = s.upper()
        example = bytes(f"'{t}'\n", encoding='utf8')
        prefix = bytes(f"Input: Uppercase the string '{s}'.\nOutput: ", encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/string/upper",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item
                       
def string_sort_feeder():
    while True:
        n = random.randint(1, string_len_level)
        s = generate_random_alphabet_string(n)
        t = ''.join(sorted(s))
        example = bytes(f"'{t}'\n", encoding='utf8')
        prefix = bytes(f"Input: Sort the string '{s}'.\nOutput: ", encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/string/sort",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item
                       
def string_len_feeder():
    while True:
        n = random.randint(0, string_len_level)
        s = generate_random_alphabet_string(n)
        t = str(len(s))
        example = bytes(str(len(s))+'\n', encoding='utf8')
        prefix = bytes(f"Input: Count the number of characters in the string '{s}'.\nOutput: ", encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/string/len",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item
        
def string_getitem_feeder():
    while True:
        n = random.randint(1, string_len_level)
        s = generate_random_alphabet_string(n)
        idx = random.randint(0,n-1)
        t = s[idx]
        example = bytes(f"'{t}'\n", encoding='utf8')
        prefix = bytes(f"Input: Let s be '{s}'. What is s[{idx}]?\nOutput: ", encoding='utf8')
        mask, training_bytes = instruction_example(prefix, example)
        training_item = {
            "source": "core/string/getitem",
            "bytes": training_bytes,
            "mask": mask
        }
        yield training_item
        
# idea for shuffle: shuffle it, and then do count structures on original and shuffled to show they are same
# or give a permutation of indices, then apply it
# def string_shuffle_feeder():
#     while True:
#         n = random.randint(1, 64)
#         s = generate_random_alphabet_string(n)
#         t = ''.join(random.shuffle(s))
#         example = bytes(f"'{t}'", encoding='utf8')
#         prefix = bytes(f"Input: Shuffle the string '{s}'.\nOutput: ", encoding='utf8')
#         mask, training_bytes = instruction_example(prefix, example)
#         training_item = {
#             "source": "core/string/shuffle",
#             'idx': None,
#             "bytes": training_bytes,
#             "mask": mask
#         }
#         yield training_item
                  
# verify a permutation of 0...n-1

