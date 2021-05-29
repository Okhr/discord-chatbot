import datetime
import json
import os
import time
from pprint import pp
from typing import List, Tuple, Dict

DATA_PATH = "data/raw"


def json2message_list(channel_messages: dict) -> List[Dict]:
    message_list = []
    for msg in channel_messages['messages']:
        # message filtering

        if msg['content'] == "":
            continue

        message_list.append({
            'content': msg['content'],
            'id': msg['id'],
            'timestamp': time.mktime(
                datetime.datetime.strptime(msg['timestamp'][:19], "%Y-%m-%dT%H:%M:%S").timetuple()),
            'reference': msg['reference']['messageId'] if 'reference' in msg.keys() else ''
        })
    return message_list


if __name__ == '__main__':

    messages_by_channel = []

    for channel_path in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, channel_path)) as fd:
            message_dict = json.load(fd)
            msg_list = json2message_list(message_dict)
            messages_by_channel.append(msg_list)

    with open('data/messages_by_channel.json', 'w', encoding='utf8') as fd:
        json.dump(messages_by_channel, fd, indent=4, ensure_ascii=False)
