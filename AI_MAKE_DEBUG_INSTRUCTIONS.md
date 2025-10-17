# デバッグ手順

出力が期待と異なる場合、以下の手順でデバッグしてください。

## 1. デバッグモードを有効化

`flash_attn_compat.py`の先頭を編集：

```python
DEBUG_MODE = True  # Falseから変更
```

## 2. 実行してログを確認

ComfyUIでワークフローを実行すると、コンソールに以下のような情報が出力されます：

```
[DEBUG] Input shapes: q=torch.Size([256, 32, 64]), k=..., v=...
[DEBUG] cu_seqlens_q=tensor([0, 128, 256]), cu_seqlens_k=...
[DEBUG] max_seqlen: q=128, k=128
[DEBUG] Batch 0: qi=torch.Size([32, 128, 64]), ki=..., vi=...
```

## 3. 問題の特定

### ケース1: "SDPA failed" メッセージが出る
→ SDPAに互換性がない。`USE_HIGH_PRECISION = True`に変更

### ケース2: 形状エラー
→ GQA（Grouped Query Attention）の処理に問題がある可能性

### ケース3: 値が異常（nan, inf）
→ 数値精度の問題。`USE_HIGH_PRECISION = True`に変更

## 4. 精度テスト

両方の設定を試して比較：

```python
# 設定1: 高速モード
USE_HIGH_PRECISION = False
DEBUG_MODE = False

# 設定2: 高精度モード
USE_HIGH_PRECISION = True
DEBUG_MODE = False
```

## 5. 報告

問題が解決しない場合、以下の情報を報告：
- デバッグログの全文
- 使用しているモデル
- 入力画像のサイズ
- 期待される出力と実際の出力の画像
