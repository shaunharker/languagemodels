
import os

import numpy as np

class LossRecorder:
    def __init__(self, output_path):
        self.output_path = output_path

    def record(self, batch, losses):
        try:
            with open(self.output_path, 'ab') as f:
                # Write batch array to file
                np.save(f, batch)
                # Write losses array to file
                np.save(f, losses)
        except Exception as e:
            print(f'Error during recording: {e}')


class LossReader():
    def __init__(self, file_path):
        self.file_path = file_path
        self.batch_size, self.num_losses = self.establish_record_size()
        self.record_length = self.batch_size + self.num_losses
        self.total_records = self.get_total_records()

    def establish_record_size(self):
        with open(self.file_path, 'rb') as f:
            batch = np.load(f)
            losses = np.load(f)
            # use batch.nbytes and losses.nbytes to get their size in bytes
            return batch.nbytes + 128, losses.nbytes + 128
            
    def get(self, index):
        with open(self.file_path, 'rb') as f:
            f.seek(index * self.record_length)
            batch = np.load(f)
            losses = np.load(f)
            return batch, losses

    def get_last(self, num=1):
        start = max(0, self.total_records - num)
        return [self.get(i) for i in range(start, self.total_records)]
        
    def get_total_records(self):
        # get file size
        file_size = os.path.getsize(self.file_path)
        total_records = file_size // self.record_length
        return total_records
