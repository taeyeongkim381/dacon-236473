import torch
from tqdm import tqdm


def evaluate(model, test_loader, checkpoint_path, device, submission_df, save_path="submission.csv"):
    # 모델 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_probs = []  # ✅ 확률 저장 리스트

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔍 Running inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.sigmoid(logits)  # ✅ logits → 확률(0~1)
            all_probs.extend(probs.cpu().tolist())

    submission_df["generated"] = all_probs
    submission_df.to_csv(save_path, index=False)
    print(f"📁 Submission saved to: {save_path}")
