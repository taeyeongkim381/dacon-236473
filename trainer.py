import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer,
                 train_loader, valid_loader,
                 batch_size, epochs, lr, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.best_auc = 0.0

    def train(self, epoch):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc=f"🔥 Epoch {epoch + 1}/{self.epochs}"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            labels = batch["label"].float().to(self.device)

            logits = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def valid(self):
        self.model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in self.valid_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                labels = batch["label"].float().to(self.device)

                logits = self.model(input_ids, attention_mask, token_type_ids)
                all_logits.extend(logits.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        probs = torch.sigmoid(torch.tensor(all_logits, device=self.device)).cpu()
        preds = (probs >= 0.5).long().tolist()
        auc = roc_auc_score(all_labels, probs)
        acc = accuracy_score(all_labels, preds)
        print(classification_report(all_labels, preds, target_names=["Human", "AI"]))

        return auc, acc

    def training(self):
        for epoch in range(self.epochs):
            train_loss = self.train(epoch)
            auc, acc = self.valid()

            print(f"✅ Epoch {epoch+1}: Loss={train_loss:.4f}, AUC={auc:.4f}, Acc={acc:.4f}")

            if auc > self.best_auc:
                self.best_auc = auc
                model_save_path = (
                    f"bs{self.batch_size}_ep{self.epochs}_"
                    f"lr{self.lr}_rocauc{auc:.4f}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': self.best_auc,
                    'train_loss': train_loss
                }, model_save_path)
                print(f"🎉 Best model saved to {model_save_path}")

        print(f"🏁 Training complete! Best AUC: {self.best_auc:.4f}")
