import argparse
import requests
import time

import networkx as nx
from tqdm import tqdm


def get_edges(profile_id: str, friends: list, token: str, api_version: str) -> dict:
    """
    the functions first requests all mutual friends of a given vk id
    and returns a dict with mutual connection per each friend
    filtering DELETED profiles

    profile_id: user_id whose friends' list is being parsed
    friends: list of dicts from requests.get(url).json()['response']['items']
    token: access token for VK API
    api_version: current VK API version

    returns: dictionary with friend_ids as keys and mutuals'ids as values
    """
    method = 'friends.getMutual'
    mutuals = {}
    for friend in tqdm(friends):
        friend_id = friend['id']
        prefix = 'https://api.vk.com/method'
        url = f'{prefix}/{method}?source_uid={profile_id}&target_uid={friend_id}&access_token={token}&v={api_version}'
        friends_ids = requests.get(url).json().get('response')
        if friends_ids != None:
            # filter out deleted profiles
            mutuals[friend_id] = friends_ids
        # wait to prevent exceeding the nemuber of requests per second
        time.sleep(3)
    return mutuals

def build_graph(friends: list, mutuals: dict) -> nx.classes.graph.Graph:
    """
    build a graph from list of friends' vk profiles and mutual connections

    friends: list of dicts by requests.get(url).json()['response']['items']
    mutuals: dict, where key is a profile_id and value is a list of mutual friends' ids

    returns a networkx Graph with nodes indexed from 0 to len(friends)
    Each node has a set atrtibutes
    """
    friends_ids = [friend['id'] for friend in friends]
    G = nx.Graph()
    G.add_nodes_from(range(len(friends_ids)))

    for idx in tqdm(friends_ids):
        node_id = friends_ids.index(idx)
        G.nodes[node_id]['vk_id'] = idx
        G.nodes[node_id]['first_name'] = friends[node_id]['first_name']
        G.nodes[node_id]['last_name'] = friends[node_id]['last_name']
        G.nodes[node_id]['gender'] = friends[node_id]['sex']
        G.nodes[node_id]['relation'] = friends[node_id].get('relation')
        G.nodes[node_id]['city'] = friends[node_id].get('city', {}).get('title')
        G.nodes[node_id]['country'] = friends[node_id].get('country', {}).get('title')
        G.nodes[node_id]['schools'] = friends[node_id].get('schools')
        G.nodes[node_id]['universities'] = friends[node_id].get('universities')
        G.nodes[node_id]['career'] = friends[node_id].get('career')
        idx_mutuals = mutuals.get(idx)
        if idx_mutuals != None:
            edges = [(node_id, friends_ids.index(friend_id)) for friend_id in idx_mutuals]
            G.add_edges_from(edges)

    return G

def main(user_id: str, token: str, api_version, filename: str):
    """
    this function will request friends list using VK API and save it
    as Graph where nodes are people and edges are connections between friends

    user_id: VK ID around which you need to build a graph
    token: VK API access token
    api_version: necessary VK api version
    filename: the resulting graph will be saved into this file
    """
    method = 'friends.get'
    fields='sex,city,country,relation,universities,schools,tv,career,interests'
    prefix = 'https://api.vk.com/method'
    url = f'{prefix}/{method}?user_id={user_id}&fields={fields}&v={api_version}&access_token={token}'
    friends = requests.get(url)
    print("Ruquest is complete: ", friends)
    friends = friends.json()['response']['items']
    mutuals = get_edges(user_id, friends, token, api_version)
    egoG = build_graph(friends, mutuals)
    print(len(egoG.nodes) == len(friends))
    nx.write_gpickle(egoG, filename)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--vk_id",
        help="A VK id whose ego graph you want to extract",
        required=True)
    arg("--token", help="Access token for VK API", required=True)
    arg("--api_version", help="Version of VK API", default='5.81')
    arg(
        "--output", help="Filename to create and store the graph",
        default='friends.gpickle')
    args = parser.parse_args()

    main(args.vk_id, args.token, args.api_version, args.output)
