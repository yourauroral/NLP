#!/usr/bin/env python3
"""下载预训练 mLSTM 权重 (sentiment.safetensors) 到 model/ 目录。

权重约 330 MB，不随仓库分发；改为外部托管、按需下载，以保持仓库轻量。

用法：
    # 方式一：填好下方 WEIGHTS_URL 后直接运行
    python download_weights.py

    # 方式二：用环境变量覆盖下载地址（无需改文件）
    export SENTIMENT_WEIGHTS_URL="https://huggingface.co/<用户>/<仓库>/resolve/main/sentiment.safetensors"
    python download_weights.py

    # 可选：指定输出文件（默认 model/sentiment.safetensors）
    python download_weights.py model/sentiment.safetensors

已存在且有效的文件会自动跳过。
"""
import os
import sys
import urllib.error
import urllib.request

# ─────────────────────────────────────────────────────────────────────────
# TODO: 权重上传到 HuggingFace 后，把 sentiment.safetensors 的完整下载地址填到这里。
#       形如： "https://huggingface.co/<用户>/<仓库>/resolve/main/sentiment.safetensors"
#       留空时可用环境变量 SENTIMENT_WEIGHTS_URL 覆盖。
WEIGHTS_URL = ""
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_DEST = os.path.join('model', 'sentiment.safetensors')
MIN_BYTES = 300 * 1024 * 1024  # 合法权重至少几百 MB，用于挡住错误页 / 截断文件


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

    if is_valid_safetensors(dest):
        print('• 已存在且有效，跳过：%s' % dest)
        return

    if not url:
        sys.exit(
            '✗ 下载地址未配置。\n'
            '  请在 download_weights.py 中填写 WEIGHTS_URL，\n'
            '  或设置环境变量： export SENTIMENT_WEIGHTS_URL=<sentiment.safetensors 的完整地址>'
        )

    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
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
