#!/usr/bin/env python3
# ============================================================================
# MULTI-TASK LEGAL AI SYSTEM  (NaN-FIXED)
# Tasks:
# 1. Case Outcome Classification (Binary)
# 2. Legal Entity Recognition (Token-level NER)
# 3. Contract Risk Score (Regression)
#
# ROOT CAUSES OF NaN LOSS — FIXED IN THIS VERSION
# ─────────────────────────────────────────────────
# FIX-1  IL-TUR LNER uses character-span annotations stored in `spans`
#        (not `ner_tags`).  Labels were never extracted → NER_LABELS=["O"]
#        → 1-class NER head → degenerate softmax → NaN.
#        Now: spans are parsed, BIO label list is built, and each token
#        is tagged via offset_mapping.
#
# FIX-2  CJPE samples have no NER supervision (all ner_labels = -100).
#        CrossEntropyLoss(ignore_index=-100) with NO valid tokens does
#        0/0 → NaN which propagates into total_loss.
#        Now: ner_loss is set to 0.0 for batches with no valid NER tokens.
#
# FIX-3  Duplicate tokenisation in LegalDataset (was tokenised twice,
#        second encoding shadowed the first).  Single tokenise path now.
# ============================================================================

# ==================== CELL 1: IMPORTS ====================
print(" SCRIPT STARTED")

import os
import re
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
    pipeline,
)
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
)
from seqeval.metrics import f1_score as seqeval_f1
from tqdm import tqdm

# NLTK sentence tokenisation
try:
    import nltk
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception:
    def sent_tokenize(text):
        return text.split('. ')
    print("⚠ NLTK unavailable – using simple sentence split")

# OCR / PDF imports
try:
    import PyPDF2
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
    PDF_SUPPORT   = True
    print("✓ OCR and PDF support enabled")
except ImportError as e:
    OCR_AVAILABLE = False
    PDF_SUPPORT   = False
    print(f"⚠ OCR/PDF libraries missing: {e}")


# ==================== CELL 2: CONFIGURATION ====================
class Config:
    MODEL_NAME           = "prajjwal1/bert-tiny"
    MAX_LEN              = 96
    BATCH_SIZE           = 32
    EPOCHS               = 1
    LR                   = 2e-5
    DROPOUT              = 0.3
    WARMUP_STEPS         = 500
    CONFIDENCE_THRESHOLD = 0.75
    BEST_MODEL_PATH      = "best_legal_model.pt"
    OCR_DPI              = 300
    OCR_LANG             = 'eng'
    # Populated by load_and_prepare_dataset(); "O" = BIO Outside tag
    NER_LABELS: list     = ["O"]
    # int→str map for ClassLabel span labels; set by load_and_prepare_dataset
    NER_ID2LABEL: dict   = {}
    DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 70)
print("Legal BERT Judgment Prediction ")
print("=" * 70)
print(f"Device:      {Config.DEVICE}")

print(f"Batch size:  {Config.BATCH_SIZE}")
print(f"Epochs:      {Config.EPOCHS}")
print(f"OCR support: {'✓ Enabled' if OCR_AVAILABLE else '✗ Disabled'}")
print("=" * 70)


# ==================== CELL 3: PDF TEXT EXTRACTOR ====================
class PDFTextExtractor:
    def __init__(self, use_ocr=True):
        self.use_ocr = use_ocr and OCR_AVAILABLE

    def extract_text(self, pdf_path):
        print(f"  📄 {os.path.basename(pdf_path)}")
        text = self._extract_direct(pdf_path)
        if len(text.strip()) < 100 and self.use_ocr:
            print("  ⚠ Sparse text – falling back to OCR…")
            text = self._extract_with_ocr(pdf_path)
        if text.strip():
            print(f"  ✓ {len(text.split())} words extracted")
        else:
            print("  ✗ No text extracted")
        return text

    def _extract_direct(self, pdf_path):
        if not PDF_SUPPORT:
            return ""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                print(f"  📖 {len(reader.pages)} pages (direct)")
                for i, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
                    if (i + 1) % 5 == 0:
                        print(f"     page {i+1}/{len(reader.pages)}")
        except Exception as e:
            print(f"  ⚠ Direct extraction error: {e}")
        return text

    def _extract_with_ocr(self, pdf_path):
        if not OCR_AVAILABLE:
            return ""
        try:
            print("  🔍 OCR …")
            images = convert_from_path(pdf_path, dpi=Config.OCR_DPI)
            print(f"  📸 {len(images)} page-images")
            ocr_text = ""
            for i, img in enumerate(images):
                img       = img.convert('L')
                page_text = pytesseract.image_to_string(
                    img, lang=Config.OCR_LANG, config='--psm 6')
                ocr_text += page_text + "\n"
                if (i + 1) % 5 == 0:
                    print(f"     OCR {i+1}/{len(images)}")
            return ocr_text
        except Exception as e:
            print(f"  ⚠ OCR error: {e}")
            return ""


# ==================== CELL 4: LEGAL TEXT CLEANER ====================
class LegalTextCleaner:
    def __init__(self):
        self.legal_abbr = {
            r'\bplaint\.?\b': 'plaintiff',    r'\bdeft\.?\b':   'defendant',
            r'\bpetnr\.?\b':  'petitioner',   r'\bresp\.?\b':   'respondent',
            r'\bapp\.?\b':    'appellant',    r'\bc\.?\b':      'court',
            r'\bj\.?\b':      'judge',        r'\bJ\.?C\.?\b':  'Justice',
            r'\bL\.?J\.?\b':  'Lord Justice', r'\bv\.?\b':      'versus',
            r'\bvs\.?\b':     'versus',       r'\bsec\.?\b':    'section',
            r'\bchap\.?\b':   'chapter',      r'\bart\.?\b':    'article',
        }
        self.ocr_noise = [
            (r'\|',             'I'),
            (r'[\u2018\u2019]', "'"),
            (r'[\u201C\u201D]', '"'),
            (r'[\u2014\u2013]', '-'),
        ]

    def clean(self, text):
        if not text:
            return ""
        for pat, rep in self.ocr_noise:
            text = re.sub(pat, rep, text)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        for pat, exp in self.legal_abbr.items():
            text = re.sub(pat, exp, text, flags=re.IGNORECASE)
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
        return text.strip()

    def segment_sentences(self, text):
        return sent_tokenize(text)


# ==================== UTILITY ====================
def extract_decision_reason(text, prediction):
    """
    Extracts structured judicial reasoning near final decision.
    """

    # Clean excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    sentences = sent_tokenize(text)

    decision_keywords = [
        "petition is allowed",
        "petition is dismissed",
        "appeal is allowed",
        "appeal is dismissed",
        "accordingly",
        "in view of the above",
        "thus",
        "hence",
        "resultantly",
        "therefore",
    ]

    # Find index of final decision sentence
    decision_index = None
    for i, sent in enumerate(sentences):
        s = sent.lower()
        if any(k in s for k in decision_keywords):
            decision_index = i
            break

    # If decision found → extract 4 sentences before it
    if decision_index is not None:
        start = max(0, decision_index - 4)
        reasoning_block = sentences[start:decision_index + 1]
    else:
        # fallback: take last 5 meaningful sentences
        reasoning_block = sentences[-5:]

    reason_text = " ".join(reasoning_block).strip()

    # Final cleanup
    reason_text = re.sub(r'\s+', ' ', reason_text)

    return reason_text if len(reason_text) > 50 else "Clear reasoning could not be extracted."


# ==================== CELL 5: DATASET ====================
class LegalDataset(Dataset):
    """
    Unified dataset for CJPE (classification / regression) and LNER (NER).

    Each sample dict must have:
      text       : str
      label      : int          classification target (0/1)
      risk_score : float        regression target
      spans      : list | None  char-level NER spans [{start, end, label}]
    """

    def __init__(self, samples, tokenizer, max_length=128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = []
        cleaner         = LegalTextCleaner()

        # Build label→id map from the global NER_LABELS (set before this call)
        self.label2id = {lbl: i for i, lbl in enumerate(Config.NER_LABELS)}

        print(f"  Creating dataset with {len(samples)} samples…")

        for item in tqdm(samples, desc="  Processing"):
            text = item.get('text', '')
            if not text:
                continue
            text  = cleaner.clean(text)
            spans = item.get('spans') or []   # list of {start, end, label}

            # ── FIX-3: single tokenise call with offset mapping ───────────────
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors='pt',
            )
            offset_mapping = encoding.pop('offset_mapping').squeeze(0).tolist()

            # ── FIX-1: BIO token labels via character offsets ─────────────────
            if spans:
                aligned_ner = self._spans_to_bio(offset_mapping, spans)
            else:
                # No NER supervision (CJPE sample) → all -100
                aligned_ner = [-100] * max_length

            # ── Classification label ──────────────────────────────────────────
            label = item.get('label', 0)
            if isinstance(label, list):
                label = label[0] if label else 0
            label = int(label)

            # ── Regression target ─────────────────────────────────────────────
            risk_score = item.get('risk_score', 0.5)
            if isinstance(risk_score, list):
                risk_score = risk_score[0] if risk_score else 0.5
            risk_score = float(risk_score)

            # ── token_type_ids safe fallback ──────────────────────────────────
            if 'token_type_ids' in encoding:
                tti = encoding['token_type_ids'].squeeze(0)
            else:
                tti = torch.zeros(max_length, dtype=torch.long)

            self.data.append({
                'input_ids':      encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'token_type_ids': tti,
                'labels':         torch.tensor(label,       dtype=torch.long),
                'ner_labels':     torch.tensor(aligned_ner, dtype=torch.long),
                'risk_scores':    torch.tensor(risk_score,  dtype=torch.float),
            })

    def _spans_to_bio(self, offset_mapping, spans):
        """
        Convert character-level spans to per-subword-token BIO label IDs.

        Handles both string labels ('COURT') and integer ClassLabel IDs (0, 1, 2…).
        Integer IDs are resolved to string names via Config.NER_ID2LABEL.

        Rules:
          offset (0, 0)                             → -100  (special / padding)
          token overlaps span AND tok_start==sp_start  → B-<label>
          token overlaps span AND tok_start > sp_start → I-<label>
          no overlap                                 → O  (id = 0)
        """
        aligned = []
        for (tok_start, tok_end) in offset_mapping:
            # Special / padding tokens have offset (0, 0)
            if tok_start == 0 and tok_end == 0:
                aligned.append(-100)
                continue

            tag_id = self.label2id.get("O", 0)   # default: Outside

            for span in spans:
                sp_start = int(span.get("start", span.get("token_start", 0)))
                sp_end   = int(span.get("end",   span.get("token_end",   0)))

                # Resolve label: integer → name via schema; string → use directly
                raw_lbl = span.get("label", "O")
                if isinstance(raw_lbl, int):
                    sp_label = Config.NER_ID2LABEL.get(raw_lbl, str(raw_lbl))
                else:
                    sp_label = str(raw_lbl)

                if tok_start >= sp_start and tok_end <= sp_end and tok_end > tok_start:
                    bio_tag = f"B-{sp_label}" if tok_start == sp_start else f"I-{sp_label}"
                    tag_id  = self.label2id.get(bio_tag, self.label2id.get("O", 0))
                    break   # first matching span wins

            aligned.append(tag_id)

        return aligned

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==================== CELL 6: MODEL ====================
class MultiTaskLegalBERT(nn.Module):
    def __init__(self, num_ner_labels):
        super().__init__()
        print("\n🔧 Initializing Multi-Task LegalBERT…")
        self.bert    = AutoModel.from_pretrained(Config.MODEL_NAME)
        hs           = self.bert.config.hidden_size
        self.dropout = nn.Dropout(Config.DROPOUT)

        print("  ✓ Classification Head (binary outcome)")
        self.classifier = nn.Sequential(
            nn.Linear(hs, 256), nn.ReLU(), nn.Dropout(Config.DROPOUT), nn.Linear(256, 2))

        print(f"  ✓ NER Head ({num_ner_labels} labels)")
        self.ner_head = nn.Sequential(
            nn.Linear(hs, hs),  nn.ReLU(), nn.Dropout(Config.DROPOUT),
            nn.Linear(hs, num_ner_labels))

        print("  ✓ Regression Head (risk score)")
        self.regressor = nn.Sequential(
            nn.Linear(hs, 128), nn.ReLU(), nn.Dropout(Config.DROPOUT), nn.Linear(128, 1))

        print("✓ Model ready.")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out  = self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        seq  = self.dropout(out.last_hidden_state)
        pool = self.dropout(out.pooler_output)

        cls_logits = self.classifier(pool)
        ner_logits = self.ner_head(seq)
        risk       = self.regressor(pool)

        return {
            'classification_logits': cls_logits,
            'classification_probs':  torch.softmax(cls_logits, dim=-1),
            'classification_pred':   torch.argmax(cls_logits, dim=-1),
            'ner_logits':            ner_logits,
            'ner_probs':             torch.softmax(ner_logits, dim=-1),
            'ner_pred':              torch.argmax(ner_logits, dim=-1),
            'risk_score':            risk.squeeze(-1),
        }


# ==================== CELL 7: DATASET LOADING ====================
def load_and_prepare_dataset():
    """
    Load IL-TUR CJPE + LNER, normalise into unified sample dicts.

    FIX-1: LNER spans are character-level dicts {start, end, label}.
    We collect every unique label string → build BIO label list →
    store in Config.NER_LABELS BEFORE LegalDataset is constructed.

    Datasets have different schemas; we never call concatenate_datasets()
    across them.
    """
    print("\n📥 Loading Hugging Face datasets…")

    # ── CJPE ──────────────────────────────────────────────────────────────────
    cjpe = concatenate_datasets([
        load_dataset("Exploration-Lab/IL-TUR", "cjpe", split="single_train"),
        load_dataset("Exploration-Lab/IL-TUR", "cjpe", split="multi_train"),
    ])
    print(f"  ✓ CJPE  : {len(cjpe)} samples  | cols: {cjpe.column_names}")

    # ── LNER ──────────────────────────────────────────────────────────────────
    lner = load_dataset("Exploration-Lab/IL-TUR", "lner", split="fold_1")
    print(f"  ✓ LNER  : {len(lner)} samples  | cols: {lner.column_names}")

    # ── FIX-1: Build BIO label list ───────────────────────────────────────────
    # IL-TUR LNER stores span['label'] as a ClassLabel integer whose string
    # names live in the HF feature schema, NOT as plain strings in the data.
    # We must read .names from the schema first; fall back to scanning values.
    id2label: dict = {}   # int → str name (used later in _spans_to_bio)

    spans_feat = lner.features.get("spans")
    if spans_feat is not None:
        # Sequence[{start, end, label, ...}]  →  .feature is the inner dict schema
        inner = getattr(spans_feat, 'feature', {})
        label_schema = inner.get("label") if isinstance(inner, dict) else None
        if label_schema is not None and hasattr(label_schema, 'names'):
            # ClassLabel: names are the ground truth
            id2label   = {i: name for i, name in enumerate(label_schema.names)}
            raw_labels = set(label_schema.names)
            print(f"  ✓ Read {len(raw_labels)} NER label names from feature schema")
        else:
            # Fallback: scan data (handles plain string labels)
            raw_labels = set()
            for ex in lner:
                for span in (ex.get("spans") or []):
                    lbl = span.get("label")
                    if lbl is not None:
                        raw_labels.add(str(lbl))
            print(f"  ✓ Scanned data → {len(raw_labels)} unique NER labels")
    else:
        raw_labels = set()

    if raw_labels:
        bio_labels = ["O"]
        for lbl in sorted(raw_labels):
            bio_labels.append(f"B-{lbl}")
            bio_labels.append(f"I-{lbl}")
        Config.NER_LABELS = bio_labels
    else:
        print("  ⚠ No span labels found in LNER – NER head will be trivial (O only).")
        Config.NER_LABELS = ["O"]

    # Store id2label in Config so _spans_to_bio can resolve integer labels
    Config.NER_ID2LABEL = id2label

    print("\n✓ NER Labels:")
    for i, lbl in enumerate(Config.NER_LABELS):
        print(f"  {i}: {lbl}")

    # ── Normalise CJPE ────────────────────────────────────────────────────────
    text_col     = "text" if "text" in cjpe.column_names else cjpe.column_names[0]
    cjpe_samples = [
        {
            "text":       str(ex.get(text_col, "")),
            "label":      int(ex.get("label", 0)),
            "spans":      None,   # no NER supervision
            "risk_score": 0.5,
        }
        for ex in cjpe
    ]

    # ── Normalise LNER ────────────────────────────────────────────────────────
    lner_samples = [
        {
            "text":       str(ex.get("text", "")),
            "label":      0,                        # no classification target
            "spans":      ex.get("spans") or [],    # raw char-level span list
            "risk_score": 0.5,
        }
        for ex in lner
    ]
    # 🔥 Oversample NER data (VERY IMPORTANT)
    lner_samples = lner_samples * 200

    # ── Combine & split ───────────────────────────────────────────────────────
    combined = cjpe_samples + lner_samples
    rng      = np.random.default_rng(42)
    rng.shuffle(combined)

    split   = int(len(combined) * 0.8)
    train_s = combined[:split]
    val_s   = combined[split:]

    print(f"\n  Total : {len(combined)}")
    print(f"  Train : {len(train_s)}")
    print(f"  Val   : {len(val_s)}")
    return {"train": train_s, "validation": val_s}


# ==================== CELL 8: TRAINING ====================
def train_model(model, train_loader, val_loader):
    print("\n🚀 Starting Multi-Task Training…")
    print("=" * 70)

    cls_loss_fn = nn.CrossEntropyLoss()
    # Use reduction='sum' so we can normalise manually only over valid tokens.
    # FIX-2: guard against batches where ALL tokens are -100 (pure CJPE).
    ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    reg_loss_fn = nn.MSELoss()

    optimizer   = AdamW(model.parameters(), lr=Config.LR, weight_decay=0.01)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(Config.WARMUP_STEPS, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val_f1  = 0.0
    best_val_mse = float('inf')

    for epoch in range(Config.EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")

        for batch in pbar:
            input_ids      = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            token_type_ids = batch['token_type_ids'].to(Config.DEVICE)
            labels         = batch['labels'].to(Config.DEVICE)
            ner_labels     = batch['ner_labels'].to(Config.DEVICE)
            risk_scores    = batch['risk_scores'].to(Config.DEVICE).float()

            outputs = model(input_ids, attention_mask, token_type_ids)

            cls_loss = cls_loss_fn(outputs['classification_logits'], labels)

            # ── FIX-2: NaN-safe NER loss ──────────────────────────────────────
            n_valid = (ner_labels != -100).sum().item()
            if n_valid > 0:
                ner_loss = ner_loss_fn(
                    outputs['ner_logits'].view(-1, outputs['ner_logits'].size(-1)),
                    ner_labels.view(-1),
                ) / n_valid                   # manual mean over valid tokens
            else:
                # All tokens masked → skip NER gradient for this batch
                ner_loss = torch.tensor(0.0, device=Config.DEVICE)

            reg_loss   = reg_loss_fn(outputs['risk_score'], risk_scores)
            total_loss = cls_loss + (5 * ner_loss) + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += total_loss.item()
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'cls':  f"{cls_loss.item():.4f}",
                'ner':  f"{ner_loss.item():.4f}",
                'reg':  f"{reg_loss.item():.4f}",
            })

        avg_train = total_train_loss / len(train_loader)
        val_m     = evaluate_model(model, val_loader)

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}:")
        print(f"  Train loss  : {avg_train:.4f}")
        print(f"  Val cls acc : {val_m['accuracy']:.4f}  F1 : {val_m['f1']:.4f}")
        print(f"  Val NER F1  : {val_m['ner_f1']:.4f}")
        print(f"  Val MSE     : {val_m['mse']:.4f}")

        if (val_m['f1'] > best_val_f1 or
                (val_m['f1'] == best_val_f1 and val_m['mse'] < best_val_mse)):
            best_val_f1  = val_m['f1']
            best_val_mse = val_m['mse']
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  ✓ Best model saved  (F1={best_val_f1:.4f}  MSE={best_val_mse:.4f})")

        print("-" * 70)

    print("\n✓ Training complete.")
    return model


# ==================== CELL 9: EVALUATION ====================
def evaluate_model(model, data_loader):
    model.eval()
    all_cls_preds, all_cls_labels  = [], []
    all_ner_preds, all_ner_labels  = [], []
    all_reg_preds, all_reg_targets = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inp  = batch['input_ids'].to(Config.DEVICE)
            amsk = batch['attention_mask'].to(Config.DEVICE)
            tti  = batch['token_type_ids'].to(Config.DEVICE)
            out  = model(inp, amsk, tti)

            all_cls_preds.extend(out['classification_pred'].cpu().numpy())
            all_cls_labels.extend(batch['labels'].cpu().numpy())

            all_reg_preds.extend(out['risk_score'].cpu().numpy())
            all_reg_targets.extend(batch['risk_scores'].cpu().numpy())

            ner_p = out['ner_pred'].cpu().numpy()
            ner_t = batch['ner_labels'].cpu().numpy()
            for preds, trues in zip(ner_p, ner_t):
                p_seq, t_seq = [], []
                for p, t in zip(preds, trues):
                    if t != -100:
                        p_seq.append(Config.NER_LABELS[p]
                                     if p < len(Config.NER_LABELS) else 'O')
                        t_seq.append(Config.NER_LABELS[t]
                                     if t < len(Config.NER_LABELS) else 'O')
                if p_seq:
                    all_ner_preds.append(p_seq)
                    all_ner_labels.append(t_seq)

    accuracy         = accuracy_score(all_cls_labels, all_cls_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_cls_labels, all_cls_preds, average='weighted', zero_division=0)
    ner_f1 = seqeval_f1(all_ner_labels, all_ner_preds) if all_ner_labels else 0.0
    mse    = mean_squared_error(all_reg_targets, all_reg_preds)

    return dict(accuracy=accuracy, precision=prec, recall=rec,
                f1=f1, ner_f1=ner_f1, mse=mse)


# ==================== CELL 10: PRINT METRICS ====================
def print_evaluation_metrics(m):
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION METRICS")
    print("=" * 70)
    print("\n⚖️  Classification (Case Outcome):")
    print(f"   Accuracy : {m['accuracy']:.4f}")
    print(f"   Precision: {m['precision']:.4f}")
    print(f"   Recall   : {m['recall']:.4f}")
    print(f"   F1-Score : {m['f1']:.4f}")
    print("\n🏷️  Named Entity Recognition:")
    print(f"   Entity-level F1: {m['ner_f1']:.4f}")
    print("\n📉  Regression (Risk Score):")
    print(f"   MSE : {m['mse']:.4f}")
    print(f"   RMSE: {np.sqrt(m['mse']):.4f}")
    print("=" * 70)


# ==================== CELL 11: PDF INFERENCE ====================
class PDFInferenceEngine:
    def __init__(self, model, tokenizer):
        self.model         = model
        self.tokenizer     = tokenizer
        self.pdf_extractor = PDFTextExtractor(use_ocr=True)
        self.text_cleaner  = LegalTextCleaner()
        print("  ✓ Loading BART Summarizer…")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
        )

    def _encode(self, text):
        enc  = self.tokenizer(text, truncation=True, padding='max_length',
                               max_length=Config.MAX_LEN, return_tensors='pt')
        ids  = enc['input_ids'].to(Config.DEVICE)
        amsk = enc['attention_mask'].to(Config.DEVICE)
        tti  = enc.get('token_type_ids',
                        torch.zeros_like(enc['input_ids'])).to(Config.DEVICE)
        return ids, amsk, tti

    def process_pdf(self, pdf_path):
        print(f"\n📄 {os.path.basename(pdf_path)}")

        print("\n  STEP 1: Text Extraction")
        raw = self.pdf_extractor.extract_text(pdf_path)
        if not raw:
            return {"error": "No text extracted"}

        print("\n  STEP 2: Cleaning")
        text = self.text_cleaner.clean(raw)
        print(f"  ✓ {len(text.split())} words")

        print("\n  STEP 2.5: Summarising")
        try:
            summary = self.summarizer(
                text[:2000], max_length=200, min_length=50, do_sample=False
            )[0]['summary_text']
            print(f"  ✓ {summary[:120]}…")
        except Exception as e:
            summary = "Summary unavailable."
            print(f"  ⚠ {e}")

        print("\n  STEP 3: Sentence Segmentation")
        sentences = self.text_cleaner.segment_sentences(text)
        print(f"  ✓ {len(sentences)} sentences")

        print("\n  STEP 4: Inference")
        words, all_probs = text.split(), []
        for i in range(0, len(words), 400):
            chunk = ' '.join(words[i:i+400])
            ids, amsk, tti = self._encode(chunk)
            self.model.eval()
            with torch.no_grad():
                all_probs.append(
                    self.model(ids, amsk, tti)['classification_probs'][0].cpu().numpy()
                )

        avg_probs  = np.mean(all_probs, axis=0)
        final_pred = int(np.argmax(avg_probs))
        confidence = float(avg_probs[final_pred])

        # NER + risk on first 1 000 chars
        ids, amsk, tti = self._encode(text[:1000])
        with torch.no_grad():
            out = self.model(ids, amsk, tti)
        ner_preds  = out['ner_pred'][0].cpu().numpy()
        tokens     = self.tokenizer.convert_ids_to_tokens(ids[0])
        risk_score = float(out['risk_score'].item())

        entities = [
            f"{tok}→{Config.NER_LABELS[p] if p < len(Config.NER_LABELS) else f'ENT_{p}'}"
            for tok, p in zip(tokens[:50], ner_preds[:50])
            if tok not in ['[CLS]', '[SEP]', '[PAD]'] and p > 0
        ]

        outcome = 'ALLOWED' if final_pred == 1 else 'DISMISSED'
        reason_text = extract_decision_reason(text, final_pred)
        print("\n  📋 RESULTS:")
        print(f"     Outcome    : {outcome}")
        print(f"     Confidence : {confidence:.2%}  "
              f"{'✓ High' if confidence >= Config.CONFIDENCE_THRESHOLD else '⚠ Low'}")
        print(f"     Risk Score : {risk_score:.4f}")
        print(f"     Entities   : {len(entities)}")
        print(f"     Reasoning  : {reason_text}")

        return {
            'summary':        summary,
            'classification': dict(prediction=outcome, label=final_pred,
                                   confidence=confidence,
                                   reliable=confidence >= Config.CONFIDENCE_THRESHOLD),
            'regression':     dict(risk_score=risk_score),
            'ner':            dict(entities=entities[:10], count=len(entities)),
            'reason': reason_text, 
            'text_stats':     dict(words=len(words), sentences=len(sentences)),
        }


# ==================== CELL 12: MAIN ====================
def main():
    print("\n" + "=" * 70)
    print("STEP 1: TRAINING PHASE")
    print("=" * 70)

    dataset_dict = load_and_prepare_dataset()      # sets Config.NER_LABELS
    tokenizer    = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    print("\n🔄 Creating PyTorch datasets…")
    train_ds = LegalDataset(dataset_dict['train'],      tokenizer, Config.MAX_LEN)
    val_ds   = LegalDataset(dataset_dict['validation'], tokenizer, Config.MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"✓ Training batches   : {len(train_loader)}")
    print(f"✓ Validation batches : {len(val_loader)}")

    # Build model AFTER Config.NER_LABELS is finalised by load_and_prepare_dataset
    model = MultiTaskLegalBERT(num_ner_labels=len(Config.NER_LABELS)).to(Config.DEVICE)
    model = train_model(model, train_loader, val_loader)

    print("\n" + "=" * 70)
    print("STEP 2: FINAL EVALUATION")
    print("=" * 70)
    print_evaluation_metrics(evaluate_model(model, val_loader))

    print("\n" + "=" * 70)
    print("STEP 3: LOADING BEST WEIGHTS")
    print("=" * 70)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE))
        print(f"✓ Loaded: {Config.BEST_MODEL_PATH}")
    else:
        print("⚠ Best weights not found – using current model")

    print("\n" + "=" * 70)
    print("STEP 4: PDF CHECK")
    print("=" * 70)
    pdfs = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]

    if pdfs:
        print(f"✓ Found {len(pdfs)} PDF(s): {pdfs}")
        print("\n" + "=" * 70)
        print("STEP 5: PDF INFERENCE")
        print("=" * 70)
        engine = PDFInferenceEngine(model, tokenizer)
        for pdf in pdfs:
            result = engine.process_pdf(pdf)
            if 'error' in result:
                print(f"  ✗ {result['error']}")
    else:
        print("✓ No PDFs found.  Training complete – exiting gracefully.")

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
