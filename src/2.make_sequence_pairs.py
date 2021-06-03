import json
import yaml
from pprint import pp
from typing import List, Dict, Tuple


def message_lists2message_pairs(message_lists: List[List[Dict]], max_time_delta: int) -> List[Tuple]:
    message_tuples = []
    for message_list in message_lists:
        for i in range(1, len(message_list)):

            previous_message_content = message_list[i - 1]['content']
            previous_message_time = message_list[i - 1]['timestamp']
            current_message_content = message_list[i]['content']
            current_message_time = message_list[i]['timestamp']

            # check if the current message is a response to a previous one
            if message_list[i]['reference'] != "":
                for j in range(i - 1, -1, -1):
                    if message_list[j]['id'] == message_list[i]['reference']:
                        message_tuples.append((message_list[j]['content'], current_message_content))
                        break

            # if not, ensure the previous message was sent less than max_time_delta seconds before the current one
            else:
                if previous_message_time > current_message_time - max_time_delta:
                    message_tuples.append((previous_message_content, current_message_content))

    return message_tuples


if __name__ == '__main__':

    with open('config.yaml') as f:
        params = yaml.load(f.read(), Loader=yaml.CLoader)

    with open('data/messages_by_channel.json', 'r') as fd:
        msg_lists = json.load(fd)

    msg_pairs = message_lists2message_pairs(msg_lists, params['Pairs']['max_time_delta'])

    with open('data/message_pairs.json', 'w') as fd:
        json.dump(msg_pairs, fd, indent=4, ensure_ascii=False)
