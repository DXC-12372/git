# -*- coding: utf-8 -*-
"""北京景区人流查询 Agent —— 仓库快照版

数据来源优先级：
  1) 仓库内 data/scenic_snapshot.csv（由 tools/refresh_snapshot.py 在国内机器上定时刷新）
  2) 兜底：实时请求政务接口（Streamlit Cloud 在境外，通常连不上，仅作本地开发用）

这样云端部署时不再依赖跨境网络。
"""
import csv
import io
import json
import os
import re
import ssl
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import urllib3
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from requests.adapters import HTTPAdapter
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings(InsecureRequestWarning)

st.set_page_config(page_title="北京景区人流查询Agent", page_icon="🏞️", layout="centered")

# ============================ 配置 ============================
def _secret(name, default=None):
    """Streamlit Cloud 的 Secrets 会同时注入为环境变量，这里两种都兼容。"""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


DEEPSEEK_API_KEY = _secret("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
BEIJING_USERKEY = _secret("BEIJING_USERKEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_CSV = DATA_DIR / "scenic_snapshot.csv"
SNAPSHOT_META = DATA_DIR / "scenic_snapshot.json"

SNAPSHOT_MAX_AGE_HOURS = 48      # 超过这个时长就在页面上提醒
MAX_CHARS = 8000                 # 喂给模型的数据长度上限
MAX_ROWS = 100                   # 单个景区最多返回多少行

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0,
    timeout=120,
    max_retries=2,
)


# ============================ TLS 适配器 ============================
class LegacySSLAdapter(HTTPAdapter):
    """兼容政务站点的 TLS 适配器。

    实测 data.beijing.gov.cn 的 ECDHE 握手极不稳定（BAD_ECPOINT /
    SSLV3_ALERT_HANDSHAKE_FAILURE），锁定 TLS1.2 并优先使用静态 RSA 套件最稳。
    """

    def __init__(self, ciphers=None, **kwargs):
        self._ciphers = ciphers
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        if self._ciphers:
            ctx.set_ciphers(self._ciphers)
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


TLS_PROFILES = [
    ("tls12-default", "DEFAULT@SECLEVEL=0"),
    ("tls12-rsa", "AES256-SHA:AES128-SHA:DES-CBC3-SHA:@SECLEVEL=0"),
    ("legacy", "ALL:@SECLEVEL=0"),
]


def _fetch_csv_once(ciphers: str, timeout=(10, 60)) -> str:
    """用指定的 TLS 配置抓一次 CSV 文本。"""
    session = requests.Session()
    session.mount("https://", LegacySSLAdapter(ciphers))
    session.mount("http://", LegacySSLAdapter(ciphers))

    resp = session.get(
        "https://data.beijing.gov.cn/cms/web/bjdata/api/userApply.jsp",
        params={"id": "e9f52465cfd7478eb0768b9978623e2d85427", "key": BEIJING_USERKEY},
        timeout=timeout,
        verify=False,
    )
    resp.raise_for_status()

    try:
        info = resp.json()
    except ValueError:
        raise RuntimeError(f"接口未返回JSON，请检查 userKey。原始返回：{resp.text[:300]}")

    if str(info.get("code")) != "0":
        raise RuntimeError(f"接口返回异常：{info}")

    file_url = (info.get("result") or {}).get("address")
    if not file_url:
        raise RuntimeError(f"未获取到文件下载地址：{info}")

    file_resp = session.get(file_url, timeout=(15, 180), verify=False)
    file_resp.raise_for_status()
    file_resp.encoding = file_resp.apparent_encoding or "gbk"
    return file_resp.text


def download_csv_live() -> str:
    """按 TLS 配置从新到旧逐个降级尝试，每种重试 2 次。"""
    if not BEIJING_USERKEY:
        raise RuntimeError("未配置 BEIJING_USERKEY，无法访问实时接口")
    errors = []
    for label, ciphers in TLS_PROFILES:
        for attempt in range(1, 3):
            try:
                return _fetch_csv_once(ciphers)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{label}#{attempt} {type(e).__name__}: {e}")
    raise RuntimeError("实时接口全部失败：" + " | ".join(errors[-4:]))


# ============================ 快照读取 ============================
@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot():
    """读取仓库内的快照，返回 (csv文本, 元数据)。"""
    if not SNAPSHOT_CSV.exists():
        raise FileNotFoundError(f"仓库中没有快照文件：{SNAPSHOT_CSV.name}")
    meta = {}
    if SNAPSHOT_META.exists():
        try:
            meta = json.loads(SNAPSHOT_META.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    text = SNAPSHOT_CSV.read_text(encoding="utf-8-sig", errors="replace")
    return text, meta


def snapshot_status():
    """返回 (是否存在, 快照时间, 距今小时数)。"""
    if not SNAPSHOT_CSV.exists():
        return False, None, None
    fetched_at = None
    if SNAPSHOT_META.exists():
        try:
            raw = json.loads(SNAPSHOT_META.read_text(encoding="utf-8")).get("fetched_at")
            if raw:
                fetched_at = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except Exception:
            fetched_at = None
    if fetched_at is None:
        fetched_at = datetime.fromtimestamp(SNAPSHOT_CSV.stat().st_mtime)
    age = (datetime.now() - fetched_at).total_seconds() / 3600
    return True, fetched_at, age


def _filter_csv(content: str, scenic_name: str) -> str:
    """只保留目标景区相关行，并做长度上限保护。"""
    name = (scenic_name or "").strip()
    try:
        reader = list(csv.reader(io.StringIO(content)))
    except Exception:
        reader = []

    if not reader:
        return (content or "")[:MAX_CHARS]

    header, body = reader[0], reader[1:]
    matched = [r for r in body if any(name in cell for cell in r)] if name else body
    if not matched:
        return ""
    matched = matched[:MAX_ROWS]
    result = "\n".join([",".join(header)] + [",".join(r) for r in matched])

    if not result.strip():
        return ""

    if len(result) > MAX_CHARS:
        result = result[:MAX_CHARS] + f"\n...（内容过长已截断，原始共 {len(result)} 字符）"
    return result


# ============================ 工具 ============================
@tool
def get_beijing_scenic_data(scenic_name: str = "") -> str:
    """获取北京市重点景区饱和指数数据。可选参数 scenic_name 为景区名称（如“朝阳公园”），
    传入后只返回该景区相关的数据行。数据来自仓库内的定时快照。"""
    source = "仓库快照"
    meta = {}
    try:
        content, meta = load_snapshot()
    except Exception as snap_err:
        source = "实时接口"
        try:
            content = download_csv_live()
        except Exception as live_err:
            return (f"数据获取失败。快照不可用（{snap_err}）；"
                    f"实时接口也不可用（{live_err}）。请让用户先运行 tools/refresh_snapshot.py 更新快照。")

    if not content or len(content.strip()) < 10:
        return "数据文件内容为空。"

    text = _filter_csv(content, scenic_name)
    if not text.strip():
        return f"未找到景区“{scenic_name}”的数据。"

    fetched = meta.get("fetched_at") or "未知"
    return f"数据来源：{source}｜数据快照时间：{fetched}（历史数据，不是当前时刻）\n{text}"


# ============================ Agent ============================
def get_current_time():
    cn = timezone(timedelta(hours=8))
    now = datetime.now(cn)
    now = datetime.now()
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    return f"当前时间是 {now:%Y-%m-%d %H:%M:%S}，今天是 {weekday}"


time_info = get_current_time()

agent = create_agent(
    model=llm,
    tools=[get_beijing_scenic_data],
    system_prompt=(
        f"当前时间是 {time_info}。"
        "你是北京景区人流预测助手。收到用户提问的景区后，调用 get_beijing_scenic_data 工具，"
        "如果获取数据失败，不要放弃，请调用 get_published_scenic_data 工具。并说明你调用的工具"
        "并把用户提到的景区名称作为 scenic_name 参数传入，只获取该景区相关数据。"
        "你获取的数据是以前的数据(时间不是当前时间)"
       """
        回答分两部分
        第一部分：基于返回的数据预测【从当前时间开始】将来24小数内和七天内的人流情况，最后给出整体出行建议。
        第二部分：基于预测的人流情况，输出两段严格JSON，**JSON只放预测数据，不要任何解释**。
                第一段（放在【DATA:】标记后面）：预测从今天当前时刻往后连续24小时的景区饱和度，数组内每个对象包含
                time: 时间，格式%Y-%m-%d %H:%M；人流饱和度：0~100数字。
                示例：[{"time":"2026-10-18 09:00","人流饱和度":30},{"time":"2026-10-18 10:00","人流饱和度":45}]
                第二段（放在【WEEK:】标记后面）：预测从今天开始往后连续7天、每天的整体饱和度，数组内每个对象包含
                date: 日期，格式%Y-%m-%d；人流饱和度：0~100数字。
                示例：[{"date":"2026-10-18","人流饱和度":40},{"date":"2026-10-19","人流饱和度":55}]
        """
    ),
)


# ============================ 结果解析 ============================
def extract_forecast_json(text: str, marker: str = "DATA"):
    """从模型回答中提取指定标记(如DATA/WEEK)后面的预测JSON数组，返回 list 或 None。"""
    if not text:
        return None
    match = re.search(rf"【{marker}:】\s*(\[.*?\])", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, list) and data else None
    except json.JSONDecodeError:
        return None


def render_forecast_chart(forecast: list, title: str = "人流饱和度预测",
                          x_keys=("time", "时间", "date", "日期")):
    """把预测数据绘制成折线图并输出到 Streamlit。"""
    df = pd.DataFrame(forecast)
    time_col = next((c for c in df.columns if any(k in c.lower() or k in c for k in x_keys)), df.columns[0])
    value_col = next((c for c in df.columns if "饱和" in c or "人流" in c or "index" in c.lower()), df.columns[-1])

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    fig = px.line(df, x=time_col, y=value_col, markers=True, title=title)
    fig.update_layout(xaxis_title="时间", yaxis_title="人流饱和度(0-100)",
                      height=420, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


# ============================ 页面 ============================
st.title("北京景区人流查询Agent")

ok, snap_time, age = snapshot_status()
if ok:
    st.caption(f"数据快照时间：{snap_time:%Y-%m-%d %H:%M}（约 {age:.0f} 小时前）")
    if age > SNAPSHOT_MAX_AGE_HOURS:
        st.warning(f"快照已 {age:.0f} 小时未更新，建议在本机运行 tools/refresh_snapshot.py 后再 push。")
else:
    st.warning("仓库中没有 data/scenic_snapshot.csv，将尝试实时访问政务接口（云端通常不可用）。")

user_input = st.text_input("请输入查询的景点：")

if st.button("查询") and user_input:
    with st.spinner("思考中..."):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            answer = result["messages"][-1].content
        except Exception as e:  # noqa: BLE001
            st.error(f"调用失败：{type(e).__name__}: {e}")
            st.stop()

        text_part = re.split(r"【DATA:】|【WEEK:】", answer)[0]
        st.markdown(text_part)

        hourly = extract_forecast_json(answer, marker="DATA")
        if hourly:
            st.subheader("未来24小时人流预测")
            render_forecast_chart(hourly, title="未来24小时人流饱和度预测")
            with st.expander("查看24小时原始预测数据"):
                st.dataframe(pd.DataFrame(hourly))
        else:
            st.warning("未能解析出24小时预测数据。")

        weekly = extract_forecast_json(answer, marker="WEEK")
        if weekly:
            st.subheader("未来七天人流预测")
            render_forecast_chart(weekly, title="未来七天每日人流饱和度预测")
            with st.expander("查看七天原始预测数据"):
                st.dataframe(pd.DataFrame(weekly))
        else:
            st.warning("未能解析出七天预测数据。")
