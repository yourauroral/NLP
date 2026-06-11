#!/usr/bin/env python3
"""下载预训练 mLSTM 权重到 model/ 目录。

权重不随仓库分发（约 330 MB），改为外部托管、按需下载，以保持仓库轻量。

用法：
    # 方式一：填好下方的 BASE_URL 后直接运行
    python download_weights.py

    # 方式二：用环境变量覆盖下载根地址（无需改文件）
    export SENTIMENT_WEIGHTS_URL="https://huggingface.co/<用户>/<仓库>/resolve/main"
    python download_weights.py

    # 可选：指定输出目录（默认 model/）
    python download_weights.py model

脚本会把 0.npy ~ 14.npy 下载到目标目录；已存在且有效的文件会自动跳过。
"""
import os
import sys
import urllib.error
import urllib.request

# ─────────────────────────────────────────────────────────────────────────
# TODO: 权重上传到 HuggingFace 后，把“下载根地址”填到这里（不要带末尾斜杠）。
#       形如： "https://huggingface.co/<用户>/<仓库>/resolve/main"
#       脚本会在其后拼接 /0.npy … /14.npy。
#       留空时可用环境变量 SENTIMENT_WEIGHTS_URL 覆盖。
BASE_URL = ""
# ─────────────────────────────────────────────────────────────────────────

NUM_SHARDS = 15  # 权重文件为 0.npy ~ 14.npy

# 已知文件大小（字节），仅作完整性提示。
# 相同架构 + 相同 dtype 重训练后大小不变；仅当改了 shape/dtype 才会不同。
EXPECTED_SIZES = {
    0: 65616, 1: 4194384, 2: 67108944, 3: 67108944, 4: 67108944,
    5: 67108944, 6: 1048656, 7: 67108944, 8: 65616, 9: 65616,
    10: 65616, 11: 16464, 12: 16464, 13: 4194384, 14: 1104,
}

NPY_MAGIC = b'\x93NUMPY'  # 合法 .npy 文件的起始魔数


def is_valid_npy(path):
    """文件存在且以 NumPy 魔数开头即视为有效。

    用于跳过已下载文件，并挡住“下载到 HTML 错误页”这类常见失败。
    """
    try:
        with open(path, 'rb') as f:
            return f.read(len(NPY_MAGIC)) == NPY_MAGIC
    except OSError:
        return False


def download_one(url, dest):
    """流式下载单个文件并显示进度。"""
    def _hook(blocks, blocksize, total):
        if total > 0:
            pct = min(100, blocks * blocksize * 100 // total)
            sys.stdout.write('\r  %s  %3d%%' % (os.path.basename(dest), pct))
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, _hook)
    sys.stdout.write('\n')


def main():
    base = (os.environ.get('SENTIMENT_WEIGHTS_URL') or BASE_URL).rstrip('/')
    if not base:
        sys.exit(
            '✗ 下载地址未配置。\n'
            '  请在 download_weights.py 中填写 BASE_URL，\n'
            '  或设置环境变量： export SENTIMENT_WEIGHTS_URL=<下载根地址>\n'
            '  期望布局： <下载根地址>/0.npy … /14.npy'
        )

    model_dir = sys.argv[1] if len(sys.argv) > 1 else 'model'
    os.makedirs(model_dir, exist_ok=True)

    downloaded = skipped = 0
    for i in range(NUM_SHARDS):
        name = '%d.npy' % i
        dest = os.path.join(model_dir, name)

        if is_valid_npy(dest):
            print('• 跳过 %s（已存在）' % name)
            skipped += 1
            continue

        url = '%s/%s' % (base, name)
        try:
            download_one(url, dest)
        except urllib.error.HTTPError as e:
            sys.exit('✗ 下载 %s 失败：HTTP %s' % (url, e.code))
        except urllib.error.URLError as e:
            sys.exit('✗ 无法连接 %s：%s' % (url, e.reason))

        # 校验下载结果，挡住错误页 / 截断文件
        if not is_valid_npy(dest):
            os.remove(dest)
            sys.exit('✗ %s 不是有效的 .npy（可能下载到了错误页面），已删除。请检查地址。' % name)

        expected = EXPECTED_SIZES.get(i)
        actual = os.path.getsize(dest)
        if expected and actual != expected:
            print('  ⚠ %s 大小为 %d 字节，预期 %d（除非改过 shape/dtype，否则应一致）'
                  % (name, actual, expected))
        downloaded += 1

    print('\n完成：下载 %d 个，跳过 %d 个，输出目录 %s/' % (downloaded, skipped, model_dir))


if __name__ == '__main__':
    main()
