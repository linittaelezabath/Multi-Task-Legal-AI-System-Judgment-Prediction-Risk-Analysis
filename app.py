# ================================
# APP.PY (Backend + Frontend)
# ================================

print("🚀 SCRIPT STARTED")
import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from flask import render_template

app = Flask(__name__, template_folder="templates", static_folder="static")
# Import EVERYTHING from po.py
import po


# ============================================
# STEP 1: RUN TRAINING PIPELINE (LIKE po.py)
# ============================================

print("\n" + "=" * 70)
print("MULTI-TASK LEGAL AI SYSTEM  (Inference Mode)")
print("=" * 70)
print(f"Device:      {po.Config.DEVICE}")
print(f"OCR support: ✓ Enabled")
print("=" * 70)


print("\n" + "=" * 70)
print("STEP 1: LOADING DATASET (to build NER labels)")
print("=" * 70)

dataset_dict = po.load_and_prepare_dataset()

tokenizer = AutoTokenizer.from_pretrained(po.Config.MODEL_NAME)

train_ds = po.LegalDataset(dataset_dict['train'], tokenizer, po.Config.MAX_LEN)
val_ds   = po.LegalDataset(dataset_dict['validation'], tokenizer, po.Config.MAX_LEN)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=po.Config.BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=po.Config.BATCH_SIZE,
    shuffle=False
)

# Build model AFTER labels are ready
model = po.MultiTaskLegalBERT(
    num_ner_labels=len(po.Config.NER_LABELS)
).to(po.Config.DEVICE)


# ============================================
# STEP 2: TRAIN MODEL (LIKE po.py)
# ============================================

model = po.train_model(model, train_loader, val_loader)

print("\n" + "=" * 70)
print("STEP 2: FINAL EVALUATION")
print("=" * 70)

metrics = po.evaluate_model(model, val_loader)
po.print_evaluation_metrics(metrics)


# ============================================
# STEP 3: LOAD BEST WEIGHTS
# ============================================

print("\n" + "=" * 70)
print("STEP 3: LOADING BEST WEIGHTS")
print("=" * 70)

if os.path.exists(po.Config.BEST_MODEL_PATH):
    model.load_state_dict(
        torch.load(po.Config.BEST_MODEL_PATH,
                   map_location=po.Config.DEVICE)
    )
    print(f"✓ Loaded: {po.Config.BEST_MODEL_PATH}")
else:
    print("⚠ Best weights not found – using current model")


# ============================================
# STEP 4: START FLASK FRONTEND
# ============================================

print("\n" + "=" * 70)
print("STEP 4: STARTING FRONTEND SERVER")
print("=" * 70)

app = Flask(__name__)

# Create inference engine
engine = po.PDFInferenceEngine(model, tokenizer)




@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(".", file.filename)
    file.save(filepath)

    result = engine.process_pdf(filepath)
    return jsonify(result)


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    print("\n✓ Backend running at http://127.0.0.1:5000")
    app.run(debug=False)