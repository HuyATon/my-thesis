import torch
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument('--repo_path', type=str, required=True, default='.', help='Path to the repository folder.')
args.add_argument('--file', type=str, default=None, help='Checkpoint file name.')

def check_saved_checkpoints(repo_path: str):
    files = sorted(os.listdir(repo_path))
    filter(lambda file: file.endswith('.pth'), files)
    valids = 0
    for checkpoint in files:
        path = os.path.join(repo_path, checkpoint)
        if check_individual(path):
            valids += 1
    print('=' * 30)
    print(f'Summary: {valids}/{len(files)} is OK.')
    
def check_individual(path: str):
    try:
        data = torch.load(path, map_location='cpu')
        print(f'{path} is OK! (keys: {list(data.keys())})')
        return True
    except Exception as e:
        print(f'{path} is corrupted! (Error: {e})')
        return False

if __name__ == '__main__':
    args = args.parse_args()
    if args.file:
        check_individual(os.path.join(args.repo_path, args.file))
    else:
        check_saved_checkpoints(args.repo_path)
    