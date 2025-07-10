import argparse
import torch
from transformers import AutoTokenizer
from transformers import logging
from model import KoELECTRABinaryClassifier
from trainer import Trainer
from data import get_df, split_df, get_datasets, get_loaders
from eval import evaluate  # ✅ 추가

logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "train.csv"
    test_path = "test.csv"
    submission_path = "sample_submission.csv"

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    train_df, test_df, submission_df = get_df(train_path, test_path, submission_path)
    train_df, val_df = split_df(train_df, valid_ratio=args.valid_ratio)
    train_dataset, val_dataset, test_dataset = get_datasets(train_df, val_df, test_df, tokenizer)
    train_loader, valid_loader, test_loader = get_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = KoELECTRABinaryClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    trainer.training()

    # ✅ 학습 후 best 모델 경로 구성
    best_model_path = (
        f"bs{args.batch_size}_ep{args.epochs}_"
        f"lr{args.lr}_rocauc{trainer.best_auc:.4f}.pt"
    )

    # ✅ 추론 및 저장
    evaluate(
        model=model,
        test_loader=test_loader,
        checkpoint_path=best_model_path,
        device=device,
        submission_df=submission_df,
        save_path="submission.csv"
    )


if __name__ == "__main__":
    main()
