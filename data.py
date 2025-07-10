import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def tokenize(self, text):
        return self.tokenizer(
            text,
            is_split_into_words=False,  # ✅ 문장이므로 False로 수정
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=False,  # 필요 없다면 False로 성능 최적화
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        if self.is_train:
            text = str(self.df.iloc[idx]["full_text"])
            label = int(self.df.iloc[idx]["generated"])
            encoding = self.tokenize(text)
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "token_type_ids": encoding["token_type_ids"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.float)  # ⬅️ float 처리
            }
        else:
            text = str(self.df.iloc[idx]["paragraph_text"])
            encoding = self.tokenize(text)
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "token_type_ids": encoding["token_type_ids"].squeeze(0),
            }


def get_df(train_path, test_path, submission_path):
    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    test_df = pd.read_csv(test_path, encoding='utf-8-sig')
    submission_df = pd.read_csv(submission_path, encoding='utf-8-sig')

    return train_df, test_df, submission_df


def split_df(train_df, valid_ratio=0.2):
    train_df, valid_df = train_test_split(train_df, test_size=valid_ratio, stratify=train_df["generated"], random_state=42)

    return train_df, valid_df


def get_datasets(train_df, valid_df, test_df, tokenizer):
    train_dataset = CustomDataset(train_df, tokenizer, is_train=True)
    valid_dataset = CustomDataset(valid_df, tokenizer, is_train=True)
    test_dataset = CustomDataset(test_df, tokenizer, is_train=False)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=4, num_workers=2):
    def make_loader(dataset, shuffle):
            loader_kwargs = {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "pin_memory": True,
                "persistent_workers": num_workers > 0,
            }

            # ⚠ prefetch_factor는 num_workers > 0일 때만 허용됨
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = 2

            return DataLoader(**loader_kwargs)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False) if val_dataset is not None else None
    test_loader = make_loader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader
