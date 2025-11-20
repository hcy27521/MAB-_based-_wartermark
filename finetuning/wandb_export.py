import pandas as pd
import wandb
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="anhtu96", type=str)
    parser.add_argument("--project", default="extract_retrain", type=str)
    parser.add_argument("--metric", default="acc", type=str)
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(args.entity + "/" + args.project)

    epoch_list, metric = [], []
    run_dict = {"epoch": []}
    for run in tqdm(runs):
        if run.state == "finished":
            run_dict[run.name] = []
            for i, row in run.history().iterrows():
                run_dict[run.name].append(row[f"trigger/{args.metric}"])
            if len(run_dict['epoch']) == 0:
                run_dict['epoch'] = list(range(len(run_dict[run.name])))
    run_df = pd.DataFrame(run_dict)
    run_df.to_csv(f"wandb_{args.project}_{args.metric}.csv", index=False)