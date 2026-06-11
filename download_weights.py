#!/usr/bin/env python3
"""准备预训练 mLSTM 权重 (sentiment.safetensors)，优先用本地权重，必要时才下载。

检查顺序：
  1) 目标位置已有有效的 sentiment.safetensors —— 直接跳过；
  2) 同目录下有原始 0.npy ~ 14.npy —— 就地转换，无需下载；
  3) 以上都没有 —— 才从 WEIGHTS_URL / 环境变量指定的地址下载。

权重约 330 MB，不随仓库分发，以保持仓库轻量。

用法：
    python download_weights.py                         # 默认 model/sentiment.safetensors
    python download_weights.py model/sentiment.safetensors

下载方式（仅在本地没有任何权重时才需要）：
    # 在下方填写 WEIGHTS_URL，或用环境变量覆盖（无需改文件）：
    export SENTIMENT_WEIGHTS_URL="https://huggingface.co/<用户>/<仓库>/resolve/main/sentiment.safetensors"
    python download_weights.py
"""
import os
import sys
import urllib.error
import urllib.request

# sentiment.safetensors 的完整下载地址（HuggingFace 直链）。
# 注意用 resolve（直接返回文件）而非 blob（网页视图）。
# 可用环境变量 SENTIMENT_WEIGHTS_URL 覆盖（无需改本文件）。
WEIGHTS_URL = "https://huggingface.co/TianmoCheng/sentiment-model/resolve/main/sentiment.safetensors"
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_DEST = os.path.join('model', 'sentiment.safetensors')
MIN_BYTES = 300 * 1024 * 1024  # 合法权重至少几百 MB，用于挡住错误页 / 截断文件
NUM_SHARDS = 15                # 原始权重分片 0.npy ~ 14.npy


def is_valid_safetensors(path):
    """结构校验：确认是 safetensors 文件且体积合理。

    safetensors 布局为：8 字节小端 header 长度 + JSON 头（以 '{' 开头）+ 数据。
    这能挡住“下载到 HTML 错误页”或文件被截断这类常见失败。
    """
    try:
        size = os.path.getsize(path)
        if size < MIN_BYTES:
            return False
        with open(path, 'rb') as f:
            head = f.read(9)
        if len(head) < 9:
            return False
        header_len = int.from_bytes(head[:8], 'little')
        return 0 < header_len < size and head[8:9] == b'{'
    except OSError:
        return False


def has_npy_shards(model_dir):
    """model_dir 下是否完整存在 15 个原始 .npy 分片。"""
    return all(os.path.exists(os.path.join(model_dir, '%d.npy' % i))
               for i in range(NUM_SHARDS))


def download(url, dest):
    """流式下载并显示进度。"""
    def _hook(blocks, blocksize, total):
        done = blocks * blocksize
        if total > 0:
            sys.stdout.write('\r  %s  %3d%%  (%.0f MB)'
                             % (os.path.basename(dest), min(100, done * 100 // total),
                                done / 1024 / 1024))
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, _hook)
    sys.stdout.write('\n')


def main():
    url = os.environ.get('SENTIMENT_WEIGHTS_URL') or WEIGHTS_URL
    dest = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEST
    model_dir = os.path.dirname(dest) or '.'

    # 1) 目标位置已有有效权重 → 直接用
    if is_valid_safetensors(dest):
        print('• 已存在且有效，跳过下载：%s' % dest)
        return

    # 2) 本地有原始 .npy → 就地转换，无需下载
    if has_npy_shards(model_dir):
        print('• 检测到本地原始权重 %s/0.npy~14.npy，就地转换（无需下载）…' % model_dir)
        try:
            from convert_weights import convert
        except ImportError as e:
            sys.exit('✗ 本地转换需要 torch / safetensors：%s\n'
                     '  请先 `pip install -r requirements.txt`。' % e)
        try:
            convert(model_dir, dest)
        except (FileNotFoundError, ValueError) as e:
            sys.exit('✗ 本地转换失败：%s' % e)
        return

    # 3) 本地没有任何权重 → 才需要下载
    if not url:
        sys.exit(
            '✗ 未找到本地权重，且未配置下载地址。可任选其一：\n'
            '  • 把 sentiment.safetensors 放到 %s/；\n'
            '  • 或把原始 0.npy ~ 14.npy 放到 %s/ 后重跑本脚本（会自动转换）；\n'
            '  • 或在 download_weights.py 填写 WEIGHTS_URL，'
            '或设置 SENTIMENT_WEIGHTS_URL=<sentiment.safetensors 的完整地址> 后重跑。'
            % (model_dir, model_dir)
        )

    os.makedirs(model_dir, exist_ok=True)
    try:
        download(url, dest)
    except urllib.error.HTTPError as e:
        sys.exit('✗ 下载失败：HTTP %s（%s）' % (e.code, url))
    except urllib.error.URLError as e:
        sys.exit('✗ 无法连接：%s（%s）' % (e.reason, url))

    # 校验下载结果，挡住错误页 / 截断文件
    if not is_valid_safetensors(dest):
        if os.path.exists(dest):
            os.remove(dest)
        sys.exit('✗ 下载结果不是有效的 safetensors（可能是错误页或被截断），已删除。请检查地址。')

    print('✓ 完成：%s (%.1f MB)' % (dest, os.path.getsize(dest) / 1024 / 1024))


if __name__ == '__main__':
    main()
