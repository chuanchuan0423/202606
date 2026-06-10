#!/usr/bin/env bash
#
# install_pip_into_env.sh
#
# 只往一个【已存在】的 conda 环境里逐个安装 pip 依赖。
# 完全绕开 conda 求解器，所以不会出现 "Solving environment ... Killed"。
#
# 用法:
#   bash install_pip_into_env.sh <环境名> [env.yaml路径]
# 例:
#   bash install_pip_into_env.sh Dan env.yaml
#   bash install_pip_into_env.sh Dan /root/autodl-tmp/AdaIR/env.yaml
#
# 特性:
#   - 从 env.yaml 的 pip: 段自动解析依赖，逐个安装
#   - 记录 成功 / 失败 清单，结束打印汇总
#   - 断点续装：重跑时已成功的自动跳过，只重试失败的
#   - 用国内镜像加速，--no-cache-dir 降低内存/磁盘峰值
#
set -uo pipefail

ENV_NAME="${1:-}"
ENV_FILE="${2:-env.yaml}"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"   # 清华镜像，autodl 上更快

if [[ -z "$ENV_NAME" ]]; then
    echo "[错误] 必须指定环境名。用法: bash $0 <环境名> [env.yaml路径]"
    echo "      例: bash $0 Dan env.yaml"
    exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[错误] 找不到环境文件: $ENV_FILE"
    exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qxF "$ENV_NAME"; then
    echo "[错误] conda 环境 '$ENV_NAME' 不存在，请先 conda create 或检查名字。"
    exit 1
fi

LOG_DIR="./pip_install_logs_${ENV_NAME}"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/failed.log"
DETAIL_LOG="$LOG_DIR/detail.log"
mkdir -p "$LOG_DIR"
: > "$DETAIL_LOG"
touch "$SUCCESS_LOG" "$FAIL_LOG"

if [[ -t 1 ]]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_INFO=$'\033[36m'; C_RST=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_INFO=""; C_RST=""
fi
log_info() { echo "${C_INFO}[INFO]${C_RST} $*"; }
log_ok()   { echo "${C_OK}[ OK ]${C_RST} $*"; }
log_err()  { echo "${C_ERR}[FAIL]${C_RST} $*"; }

already_done() { grep -qxF "$1" "$SUCCESS_LOG" 2>/dev/null; }
mark_success() {
    echo "$1" >> "$SUCCESS_LOG"
    if [[ -s "$FAIL_LOG" ]]; then
        grep -vxF "$1" "$FAIL_LOG" > "$FAIL_LOG.tmp" 2>/dev/null && mv "$FAIL_LOG.tmp" "$FAIL_LOG"
    fi
}
mark_fail() { grep -qxF "$1" "$FAIL_LOG" 2>/dev/null || echo "$1" >> "$FAIL_LOG"; }

# 解析 env.yaml 的 pip: 段（不使用 awk 区间 {n}，兼容 mawk）
parse_pip_deps() {
    awk '
        /^[[:space:]]*-[[:space:]]*pip:[[:space:]]*$/ { in_pip=1; next }
        in_pip {
            if ($0 ~ /^[[:space:]]*-[[:space:]]/) {
                line=$0
                sub(/^[[:space:]]*-[[:space:]]*/, "", line)
                print line
            } else if ($0 ~ /[^[:space:]]/) {
                in_pip=0
            }
        }
    ' "$ENV_FILE"
}

install_pip_pkg() {
    local spec="$1"
    if already_done "$spec"; then
        log_ok "已装过，跳过: $spec"
        return 0
    fi
    echo "============================================================" >> "$DETAIL_LOG"
    echo "[pip] $spec  $(date '+%F %T')" >> "$DETAIL_LOG"
    log_info "安装: $spec"
    if conda run -n "$ENV_NAME" python -m pip install \
            --no-cache-dir -i "$PIP_INDEX" "$spec" >> "$DETAIL_LOG" 2>&1; then
        log_ok "$spec"
        mark_success "$spec"
        return 0
    fi
    log_err "$spec  (失败，详见 $DETAIL_LOG)"
    mark_fail "$spec"
    return 1
}

# ---------------------------------------------------------------------------
log_info "目标环境: $ENV_NAME"
log_info "环境文件: $ENV_FILE"

# 先升级 pip 本身，避免老 pip 解析新包失败
log_info "升级 pip / setuptools / wheel ..."
conda run -n "$ENV_NAME" python -m pip install --no-cache-dir -i "$PIP_INDEX" \
    -U pip setuptools wheel >> "$DETAIL_LOG" 2>&1 || \
    log_err "pip 升级失败（不影响后续，继续）"

mapfile -t PIP_DEPS < <(parse_pip_deps)
log_info "解析到 pip 依赖 ${#PIP_DEPS[@]} 个，开始逐个安装..."

for dep in "${PIP_DEPS[@]}"; do
    [[ -z "$dep" ]] && continue
    install_pip_pkg "$dep"
done

# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "                       安装结果汇总"
echo "============================================================"
SUCCESS_COUNT=$(grep -c . "$SUCCESS_LOG" 2>/dev/null); SUCCESS_COUNT=${SUCCESS_COUNT:-0}
FAIL_COUNT=$(grep -c . "$FAIL_LOG" 2>/dev/null); FAIL_COUNT=${FAIL_COUNT:-0}

log_ok "成功: $SUCCESS_COUNT 个  (清单见 $SUCCESS_LOG)"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    log_err "失败: $FAIL_COUNT 个  (清单见 $FAIL_LOG)"
    echo "------------------- 失败清单 -------------------"
    cat "$FAIL_LOG"
    echo "------------------------------------------------"
    echo "提示: 直接重跑即可，已成功的自动跳过，只重试失败的。"
    echo "      逐包详细日志在: $DETAIL_LOG"
    exit 1
else
    log_ok "全部 pip 依赖安装成功！  conda activate $ENV_NAME 即可使用"
    exit 0
fi