import json
files = [
    'proposed_run/result/qmix/RDM_proposedf/log_params.txt',
    'modelqmix_run/result/qmix/RDM_model_qmix/log_params.txt',
    'qmix_run/result/qmix/RDM_qmix/log_params.txt',
    'original_run/result/qmix/RDM_original/log_params.txt'
]
with open('episodes_check.txt', 'w') as out:
    for f in files:
        try:
            params = json.load(open(f))
            out.write(f"{f}: total_episodes = {params.get('total_episodes')}\n")
        except Exception as e:
            out.write(f"{f}: ERROR {e}\n")
