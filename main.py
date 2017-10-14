import argparse
import os
from util import *
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    
    args = parser.parse_args()

    if not os.path.exists(args.dict_path):
        dict_ = mk_dict(args.data_path)
        save_dict(dict_, args.dict_path)

    model_ = model(args)
    if args.train:
        model.train()

