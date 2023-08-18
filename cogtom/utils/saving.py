import json


def save_q_table(q_values, path):
    # convert q-table keys to string for json serialization
    q_values_str = {str(k): v.tolist() for k, v in q_values.items()}
    # save the q-table
    with open(path, "w") as f:
        json.dump(q_values_str, f)
