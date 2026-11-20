from utils import *
from models import *

def train_simple_baseline(model, name, train_loader, valid_loader, test_loader, save_path, patience=10):
    print(f"--- Training {name} ---")
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, 101):
        model.train()
        train_loss, n_train = 0.0, 0
        for t, a, v, y in train_loader:
            t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(t, a, v), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            n_train += y.size(0)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for t, a, v, y in valid_loader:
                t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
                val_loss += criterion(model(t, a, v), y).item() * y.size(0)
                n_val += y.size(0)
        val_loss /= n_val

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/100 | Train MAE: {train_loss/n_train:.4f} | Val MAE: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for t, a, v, y in test_loader:
            preds.extend(model(t.to(device), a.to(device), v.to(device)).cpu().numpy().reshape(-1))
            labels.extend(y.numpy().reshape(-1))
    np.savez(save_path, y_true=np.asarray(labels), y_pred=np.asarray(preds))
    print(f"=> {name} finished | Best Val MAE: {best_loss:.4f}\n")
    return best_loss

# 3.1 Train Early Fusion (EF-DNN)
set_seed(1115, deterministic=False)
train_loader, valid_loader, test_loader = get_feature_loaders(data, batch_size=128, seed=1115, use_generator=True)
early_model = EF-DNN().to(device)
train_simple_baseline(early_model, "Early Fusion (EF-DNN)", train_loader, valid_loader, test_loader, f"{SAVE_DIR}/early_fusion_predictions.npz", patience=10)

# 3.2 Train Late Fusion (LF-DNN)
late_model = LateFusionModel().to(device)
train_simple_baseline(late_model, "Late Fusion (LF-DNN)", train_loader, valid_loader, test_loader, f"{SAVE_DIR}/late_fusion_predictions.npz", patience=8)

# 3.3 Train Hybrid Fusion (CENET)
print("--- Training Hybrid Fusion (CENET) ---")
set_seed(42, deterministic=True)
g_cenet = torch.Generator().manual_seed(42)
cenet_train_loader = DataLoader(BertMOSIDataset(data["train"]), batch_size=16, shuffle=True, generator=g_cenet)
cenet_valid_loader = DataLoader(BertMOSIDataset(data["valid"]), batch_size=16, shuffle=False)
cenet_test_loader  = DataLoader(BertMOSIDataset(data["test"]),  batch_size=16, shuffle=False)

cenet_model = CENET(injection_index=1, dropout=0.2).to(device)
criterion_mse = nn.MSELoss()
optimizer_cenet = torch.optim.AdamW(cenet_model.parameters(), lr=2e-5, weight_decay=1e-4)
scheduler_cenet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cenet, mode="min", factor=0.5, patience=2)

def eval_cenet(loader):
    cenet_model.eval()
    loss_sum, count, preds, labels = 0.0, 0, [], []
    with torch.no_grad():
        for tb, a, v, y in loader:
            tb, a, v, y = tb.to(device), a.to(device), v.to(device), y.to(device)
            out = cenet_model(tb, a, v)
            loss_sum += criterion_mse(out, y).item() * y.size(0)
            count += y.size(0)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return loss_sum / count, np.asarray(preds), np.asarray(labels)

best_cenet_mse, best_cenet_state = float("inf"), None
for epoch in range(1, 21):
    cenet_model.train()
    train_sum, train_cnt = 0.0, 0
    for tb, a, v, y in cenet_train_loader:
        tb, a, v, y = tb.to(device), a.to(device), v.to(device), y.to(device)
        optimizer_cenet.zero_grad(set_to_none=True)
        loss = criterion_mse(cenet_model(tb, a, v), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cenet_model.parameters(), 1.0)
        optimizer_cenet.step()
        train_sum += loss.item() * y.size(0)
        train_cnt += y.size(0)

    val_mse, _, _ = eval_cenet(cenet_valid_loader)
    scheduler_cenet.step(val_mse)
    status = ""
    if val_mse < best_cenet_mse:
        best_cenet_mse = val_mse
        best_cenet_state = copy.deepcopy(cenet_model.state_dict())
        status = " [BEST]"
    print(f"Epoch {epoch:02d}/20 | Train MSE: {train_sum/train_cnt:.4f} | Val MSE: {val_mse:.4f}{status}")

cenet_model.load_state_dict(best_cenet_state)
_, valid_preds, valid_labels = eval_cenet(cenet_valid_loader)
_, cenet_predictions, cenet_labels = eval_cenet(cenet_test_loader)

np.savez(f"{SAVE_DIR}/cenet_valid_predictions.npz", y_true=valid_labels, y_pred=valid_preds)
np.savez(f"{SAVE_DIR}/cenet_predictions.npz", y_true=cenet_labels, y_pred=cenet_predictions)
print(f"=> Hybrid CENET finished | Best Val MSE: {best_cenet_mse:.4f}")

# 4.1 Early Fusion + Multi-Step CoT
print("--- Training Early Fusion + CoT ---")
set_seed(1115, deterministic=False)
train_loader, valid_loader, test_loader = get_feature_loaders(data, batch_size=128, seed=1115, use_generator=False)
early_cot_model = EarlyFusionCoT(hidden_dim=128, dropout=0.25).to(device)
criterion_huber = nn.SmoothL1Loss(beta=1.0)
opt_early_cot = torch.optim.AdamW(early_cot_model.parameters(), lr=1e-4, weight_decay=1e-3)
sch_early_cot = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_early_cot, mode="min", factor=0.5, patience=3, min_lr=1e-6)

best_val_mae, best_state, wait = float("inf"), None, 0
for epoch in range(1, 101):
    early_cot_model.train()
    train_loss, n_train = 0.0, 0
    for t, a, v, y in train_loader:
        t, a, v, target = t.to(device), a.to(device), v.to(device), y.to(device).reshape(-1)
        opt_early_cot.zero_grad(set_to_none=True)
        final_pred, step_preds, _ = early_cot_model(t, a, v)
        loss = criterion_huber(final_pred, target) + 0.10 * (sum(criterion_huber(p, target) for p in step_preds) / 3.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(early_cot_model.parameters(), 1.0)
        opt_early_cot.step()
        train_loss += loss.item() * target.size(0)
        n_train += target.size(0)

    early_cot_model.eval()
    val_loss, n_val = 0.0, 0
    with torch.no_grad():
        for t, a, v, y in valid_loader:
            t, a, v, target = t.to(device), a.to(device), v.to(device), y.to(device).reshape(-1)
            pred, _, _ = early_cot_model(t, a, v)
            val_loss += criterion_huber(pred, target).item() * target.size(0)
            n_val += target.size(0)
    val_mae = val_loss / n_val
    sch_early_cot.step(val_mae)

    status = ""
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_state = copy.deepcopy(early_cot_model.state_dict())
        wait = 0
        status = " [BEST]"
    else:
        wait += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/100 | Train Loss: {train_loss/n_train:.4f} | Val MAE: {val_mae:.4f}{status}")

    if wait >= 12:
        print(f"Early stopping at epoch {epoch}")
        break

early_cot_model.load_state_dict(best_state)
early_cot_model.eval()
preds, labels = [], []
with torch.no_grad():
    for t, a, v, y in test_loader:
        p, _, _ = early_cot_model(t.to(device), a.to(device), v.to(device))
        preds.extend(p.cpu().numpy().reshape(-1))
        labels.extend(y.numpy().reshape(-1))
np.savez(f"{SAVE_DIR}/early_fusion_cot_predictions.npz", y_true=np.asarray(labels), y_pred=np.asarray(preds))
print(f"=> Early Fusion + CoT finished | Best Val MAE: {best_val_mae:.4f}\n")

# 4.2 Late Fusion + PEAR-CoT + CMFF
print("--- Training Late Fusion + PEAR-CoT + CMFF ---")
set_seed(42, deterministic=True)
train_loader_32, valid_loader_32, test_loader_32 = get_feature_loaders(data, batch_size=32, seed=42, use_generator=False)
pear_model = PEARCoT_CMFF().to(device)
criterion_pear = nn.SmoothL1Loss(beta=0.8)
opt_pear = torch.optim.AdamW(pear_model.parameters(), lr=3e-4, weight_decay=5e-3)
sch_pear = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pear, mode="max", factor=0.5, patience=3, min_lr=1e-6)

def predict_pear(loader):
    pear_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for t, a, v, y in loader:
            out = pear_model(t.to(device), a.to(device), v.to(device))
            preds.extend(out.cpu().numpy().reshape(-1))
            labels.extend(y.numpy().reshape(-1))
    return np.asarray(preds), np.asarray(labels)

def find_best_threshold(y_true, y_pred, nonzero=False):
    mask = (y_true != 0) if nonzero else np.ones(len(y_true), dtype=bool)
    true = (y_true[mask] >= 0).astype(int)
    pred = y_pred[mask]
    best_th, best_sc = 0.0, -1.0
    for th in np.arange(-0.40, 0.401, 0.005):
        predicted = (pred >= th).astype(int)
        sc = 0.75 * accuracy_score(true, predicted) + 0.25 * f1_score(true, predicted, zero_division=0)
        if sc > best_sc:
            best_sc, best_th = sc, float(th)
    return best_th

best_score, best_pear_state, wait = -1.0, None, 0
for epoch in range(1, 81):
    pear_model.train()
    train_loss, n_train = 0.0, 0
    for t, a, v, y in train_loader_32:
        t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
        opt_pear.zero_grad(set_to_none=True)
        loss = criterion_pear(pear_model(t, a, v), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pear_model.parameters(), 1.0)
        opt_pear.step()
        train_loss += loss.item() * y.size(0)
        n_train += y.size(0)

    vp, vt = predict_pear(valid_loader_32)
    th_h = find_best_threshold(vt, vp, nonzero=False)
    th_n = find_best_threshold(vt, vp, nonzero=True)
    h_acc = accuracy_score((vt >= 0).astype(int), (vp >= th_h).astype(int))
    h_f1  = f1_score((vt >= 0).astype(int), (vp >= th_h).astype(int), zero_division=0)
    mask = vt != 0
    n_acc = accuracy_score((vt[mask] >= 0).astype(int), (vp[mask] >= th_n).astype(int))
    n_f1  = f1_score((vt[mask] >= 0).astype(int), (vp[mask] >= th_n).astype(int), zero_division=0)
    score = 0.35 * h_acc + 0.35 * n_acc + 0.15 * h_f1 + 0.15 * n_f1
    sch_pear.step(score)

    status = ""
    if score > best_score:
        best_score = score
        best_pear_state = copy.deepcopy(pear_model.state_dict())
        wait = 0
        status = " [BEST]"
    else:
        wait += 1

    if epoch % 2 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/80 | Loss: {train_loss/n_train:.4f} | Has0: {h_acc*100:.2f}% | Non0: {n_acc*100:.2f}% | Score: {score:.4f}{status}")

    if wait >= 14:
        print(f"Early stopping at epoch {epoch}")
        break

pear_model.load_state_dict(best_pear_state)
test_p, test_t = predict_pear(test_loader_32)
np.savez(f"{SAVE_DIR}/pear_cot_cmff_optimized_predictions.npz", y_true=test_t, y_pred=test_p)
print(f"=> Late Fusion + PEAR-CoT + CMFF finished | Best Val Score: {best_score:.4f}")

# Free previous neural models from GPU memory before loading Qwen2.5-7B
for var_name in ["early_model", "late_model", "cenet_model", "early_cot_model", "pear_model"]:
    if var_name in globals():
        del globals()[var_name]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {LLM_NAME}...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_tokenizer.padding_side = "left"

llm_model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.float16, device_map="auto")
llm_model.eval()
print(f"Successfully loaded {LLM_NAME}")

def make_prompt(text, audio, vision, initial_score):
    return f"""Analyze sentiment of this CMU-MOSI utterance.

Text: {str(text)[:1500]}

CENET initial sentiment score: {initial_score:.3f}
Audio intensity: {np.mean(np.abs(audio)):.3f}
Visual intensity: {np.mean(np.abs(vision)):.3f}

Sentiment scale:
-3 = very negative
0 = neutral
+3 = very positive

Consider the actual meaning, context, sarcasm and emotional tone.

Return ONLY one numerical sentiment score between -3 and +3.
Do not explain.
Do not write analysis.
Do not use Markdown.

Answer:"""

def extract_score(response):
    response = response.strip()
    for pattern in [r"SCORE\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", r"FINAL\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))"]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and np.isfinite(float(match.group(1))) and -3 <= float(match.group(1)) <= 3:
            return float(match.group(1))
    match = re.fullmatch(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*", response)
    if match and np.isfinite(float(match.group(1))) and -3 <= float(match.group(1)) <= 3:
        return float(match.group(1))
    return np.nan

def run_cot_inference(texts, audio, vision, predictions, split_name, batch_size=8):
    n = len(predictions)
    scores = np.full(n, np.nan)
    print(f"\n--- Running CoT Inference on {split_name} ({n} samples) ---")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompts = [
            llm_tokenizer.apply_chat_template(
                [{"role": "user", "content": make_prompt(texts[i], audio[i], vision[i], predictions[i])}],
                tokenize=False, add_generation_prompt=True
            )
            for i in range(start, end)
        ]
        inputs = llm_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(llm_model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = llm_model.generate(
                **inputs, max_new_tokens=16, do_sample=False, num_beams=1,
                pad_token_id=llm_tokenizer.pad_token_id, eos_token_id=llm_tokenizer.eos_token_id
            )
        for j in range(len(prompts)):
            resp = llm_tokenizer.decode(outputs[j, input_len:], skip_special_tokens=True)
            scores[start + j] = extract_score(resp)

        if end % 40 == 0 or end == n:
            print(f"{split_name}: {end}/{n} processed")

        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    valid_cnt = int(np.isfinite(scores).sum())
    print(f"{split_name} valid CoT scores: {valid_cnt}/{n} ({valid_cnt/n*100:.2f}%)")
    return scores

def fuse_predictions(cenet, cot, beta):
    fused = cenet.copy()
    valid = np.isfinite(cot)
    fused[valid] = (1 - beta) * cenet[valid] + beta * cot[valid]
    return fused

# Generate CoT scores on Validation and optimize Beta & Threshold
cot_val = run_cot_inference(data["valid"]["raw_text"], np.asarray(data["valid"]["audio"]), np.asarray(data["valid"]["vision"]), valid_preds, "VALIDATION")

best_val_acc = accuracy_score(valid_labels >= 0, valid_preds >= 0)
best_beta, best_threshold = 0.0, 0.0
val_records = []

for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
    fused = fuse_predictions(valid_preds, cot_val, beta)
    for th in np.arange(-0.30, 0.301, 0.025):
        acc = accuracy_score(valid_labels >= 0, fused >= th)
        val_records.append({"Beta": beta, "Threshold": round(float(th), 4), "Validation_Acc2": acc})
        if acc > best_val_acc:
            best_val_acc, best_beta, best_threshold = acc, beta, float(th)

pd.DataFrame(val_records).to_csv("./results/contrastive_cot_validation_results.csv", index=False)
print(f"\nOptimal Validation Config -> Beta: {best_beta}, Threshold: {best_threshold:.3f}, Val Acc-2: {best_val_acc*100:.2f}%")

# Generate CoT scores on Test and apply fixed configuration
cot_test = run_cot_inference(data["test"]["raw_text"], np.asarray(data["test"]["audio"]), np.asarray(data["test"]["vision"]), cenet_predictions, "TEST")
hybrid_test_preds = fuse_predictions(cenet_predictions, cot_test, best_beta) - best_threshold

np.savez(
    f"{SAVE_DIR}/cenet_cot_predictions.npz",
    y_true=cenet_labels, y_pred=hybrid_test_preds,
    cenet_preds=cenet_predictions, cot_scores=cot_test,
    best_beta=best_beta, best_threshold=best_threshold
)
print("=> Hybrid CENET + Contrastive CoT completed!")
