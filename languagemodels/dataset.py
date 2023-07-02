import types
import numpy as np
import torch

def utf8encode(char_sequence):
    if type(char_sequence) == types.GeneratorType:
        def stream():
            for c in char_sequence:
                for b in bytes(c, encoding='utf8'):
                    yield b
        result = stream()
    else:
        result = bytes(char_sequence, encoding='utf8')
    return result

def utf8decode(byte_sequence):
    def is_valid_utf8_byte(b):
        return b&0b11111000 != 0b11111000
    def is_payload_utf8_byte(b):
        return b&0b11000000 == 0b10000000
    def is_header_utf8_byte(b):
        return is_valid_utf8_byte(b) and not is_payload_utf8_byte(b)
    def char_width(b):
        if b&0b10000000 == 0:
            return 1
        elif b&0b11100000 == 0b11000000:
            return 2
        elif b&0b11110000 == 0b11100000:
            return 3
        elif b&0b11111000 == 0b11110000:
            return 4
        return None
    def stream():
        (word, width) = ([], 0)
        for b in byte_sequence:
            if is_header_utf8_byte(b):
                (word, width) = ([b], char_width(b))
            elif is_payload_utf8_byte(b):
                word.append(b)
            if len(word) == width:
                try:
                    yield bytes(word).decode('utf8')
                except:
                    # There are still undecodables we catch here.
                    # e.g. bytes(map(lambda x: int(x,base=2),['0b11000000', '0b10000000'])).decode('utf8') raises UnicodeDecodeError
                    pass
    if type(byte_sequence) == types.GeneratorType:
        return stream()
    else:
        return ''.join(list(stream()))

def utf8bitsencode(char_seq: str):
    return np.unpackbits(np.frombuffer(bytes(char_seq, encoding='utf-8'), dtype=np.uint8),
        bitorder='little').tolist()

def utf8bitsdecode(bits):
    result = bytes()
    idx = 0
    while idx+7 < len(bits):
        if bits[idx+7] == 0:
            result += bytes([sum(2**i * bits[idx + i] for i in range(8))])
            idx += 8
        elif idx+15 < len(bits) and  (bits[idx+5] == 0 and bits[idx+6] == 1 and bits[idx+7] == 1 and
              bits[idx+14] == 0 and bits[idx+15] == 1):
            result += bytes([sum(2**i * bits[idx + i] for i in range(8)),
                       sum(2**i * bits[idx + i + 8] for i in range(8))])
            idx += 16
        elif idx+23 < len(bits) and (bits[idx+4] == 0 and bits[idx+5] == 1 and bits[idx+6] == 1 and bits[idx+7] == 1 and
              bits[idx+14] == 0 and bits[idx+15] == 1 and
              bits[idx+22] == 0 and bits[idx+23] == 1):
            result += bytes([sum(2**i * bits[idx + i] for i in range(8)),
                       sum(2**i * bits[idx + i + 8] for i in range(8)),
                       sum(2**i * bits[idx + i + 16] for i in range(8))])
            idx += 24
        elif idx+31 < len(bits) and (bits[idx+3] == 0 and bits[idx+4] == 1 and bits[idx+5] == 1 and bits[idx+6] == 1 and bits[idx+7] == 1 and
              bits[idx+14] == 0 and bits[idx+15] == 1 and
              bits[idx+22] == 0 and bits[idx+23] == 1 and
              bits[idx+30] == 0 and bits[idx+31] == 1):
            result += bytes([sum(2**i * bits[idx + i] for i in range(8)),
                       sum(2**i * bits[idx + i + 8] for i in range(8)),
                       sum(2**i * bits[idx + i + 16] for i in range(8)),
                       sum(2**i * bits[idx + i + 24] for i in range(8))])
            idx += 32
        else:
            idx += 1
    return result.decode('utf-8')


class FastPileBytesDataset:
    def __init__(self, example_length=512, paths=None, device='cuda'):
        if paths is None:
            paths = [f"/data/thepile/00.{i}.utf8" for i in range(10)]
        self.paths = paths
        self.device = device
        self.decode = utf8decode
        self.encode = utf8encode
        self.idx = 0
        self.path_idx = 0
        self.example_length = example_length
        self.load_from_dataset()

    def load_from_dataset(self):
        example_length = self.example_length
        data = np.fromfile(self.paths[self.path_idx], dtype=np.uint8)
        n_examples = len(data) // example_length
        data = data[:n_examples * example_length]
        data = data.reshape((n_examples, example_length))
        np.random.shuffle(data)
        self.data = data
        self.path_idx += 1
        if self.path_idx == len(self.paths):
            self.path_idx = 0
        self.idx = 0

    def batch(self, batch_size, example_length):
        if example_length > self.example_length:
            raise ValueError(f"example_length of {example_length} is unsupported for this instance of FastPileBytesDataset, reconstruct")
        if self.idx + batch_size > len(self.data):
            self.load_from_dataset()
        result = torch.from_numpy(self.data[self.idx:self.idx+batch_size])[:,:example_length].long().to('cuda')
        self.idx += batch_size
        return result
    
# """
# Unused bytes in utf-8 encodings:
# [0, 2, 4, 5, 6, 11, 14, 15, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 127, 192, 193, 222, 223, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
# """