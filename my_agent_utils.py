import pandas as pd

def add_agent_id (
        agent_id,
        a,
        b,
        c
):

    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    stat['Agent'].append(agent_id)
    stat['0.025'].append(a)
    stat['0.500'].append(b)
    stat['0.975'].append(c)

    return stat

def combine_stat (
        stats
):

    res = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for s in stats:

        print(f"===> {s}")

        res['Agent'].append(*s['Agent'])
        res['0.025'].append(*s['0.025'])
        res['0.500'].append(*s['0.500'])
        res['0.975'].append(*s['0.975'])

    return pd.DataFrame().from_dict(res)
