import os
import json

filename = '01.jsonl'
output_filename_prefix = '01'
output_filename_suffix = '.utf8'
output_filename_index = 0
output_file = None
delimiter = '\n'  # replace with your desired delimiter

with open(filename, 'r') as f:
    for line in f:
        json_line = json.loads(line)
        text = json_line.get('text', '')
        encoded_text = text.encode('utf-8')

        if output_file is None:
            output_filename = f"{output_filename_prefix}.{output_filename_index}{output_filename_suffix}"
            output_file = open(output_filename, 'wb')
        elif output_file.tell() + len(encoded_text) + len(delimiter.encode('utf-8')) > 4*1024**3:  # 4GB limit
            output_file.close()
            output_filename_index += 1
            output_filename = f"{output_filename_prefix}.{output_filename_index}{output_filename_suffix}"
            output_file = open(output_filename, 'wb')

        output_file.write(encoded_text)
        output_file.write(delimiter.encode('utf-8'))

if output_file is not None:
    output_file.close()
