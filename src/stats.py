import json
import os
from pprint import pprint as pp

DATA_PATH = "../data/raw"


def count_messages_per_channel(data_path):
    total_messages = 0
    messages_per_channel_dict = {}
    for channel_path in os.listdir(data_path):
        with open(os.path.join(data_path, channel_path)) as fd:
            message_dict = json.load(fd)

        channel_name = message_dict['channel']['name']
        channel_count = int(message_dict['messageCount'])

        messages_per_channel_dict[channel_name] = channel_count
        total_messages += channel_count

    return messages_per_channel_dict


def count_messages_per_author(data_path):
    messages_per_author_dict = {}

    for channel_path in os.listdir(data_path):
        with open(os.path.join(data_path, channel_path)) as fd:
            message_dict = json.load(fd)

        for message in message_dict['messages']:
            name = message['author']['name']
            nick = message['author']['nickname']

            if name not in messages_per_author_dict.keys():
                messages_per_author_dict[name] = [nick, 1]
            else:
                count = messages_per_author_dict[name][1]
                messages_per_author_dict[name] = [nick, count + 1]

    return messages_per_author_dict


if __name__ == '__main__':
    pp(count_messages_per_author(DATA_PATH))
    pp(count_messages_per_channel(DATA_PATH))
