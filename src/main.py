import json
import os
from pprint import pprint as pp

DATA_PATH = "data"


def count_messages(data_path):
    total_messages = 0
    message_counts = []
    for channel_path in os.listdir(data_path):
        with open(os.path.join(data_path, channel_path)) as fd:
            message_dict = json.load(fd)

        channel_name = message_dict['channel']['name']
        channel_count = int(message_dict['messageCount'])

        message_counts.append((channel_name, channel_count))
        total_messages += channel_count

    for item in sorted(message_counts, key=lambda x: x[1], reverse=True):
        print(f"{item[0]} : \t{item[1]}".expandtabs(50))
    print(f"\nTotal messages : {total_messages}")

    return message_counts


if __name__ == '__main__':
    count_messages(DATA_PATH)
