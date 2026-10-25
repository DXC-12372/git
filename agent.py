import streamlit as st
import requests
import os
import io
import csv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from urllib3.exceptions import InsecureRequestWarning
from datetime import datetime
import json
import plotly.express as px
import pandas as pd
import re


requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# 配置读取：优先 Streamlit secrets（云端），其次环境变量（本机）
def _get_secret(name: str, default: str = None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)

DEEPSEEK_API_KEY = _get_secret("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
BEIJING_USERKEY = _get_secret("BEIJING_USERKEY")
PUBLISHED_DATA_URL = _get_secret(
    "PUBLISHED_DATA_URL",
    "https://raw.githubusercontent.com/DXC-12372/git/main/scenic_data.json"
)
agent_gaodekey = _get_secret("agent_gaodekey")

# 景区简称/别名 -> 知识库正式名称（方便用户用口语化简称查询）
SCENIC_ALIASES = {
    "奥森": "奥林匹克森林公园",
    "奥园": "奥林匹克森林公园",
    "北动": "北京动物园",
    "八达岭": "八达岭长城",
    "长城": "八达岭长城",
    "南锣": "南锣鼓巷",
    "后海": "什刹海",
    "前海": "什刹海",
}


llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0
)


class LegacySSLAdapter(HTTPAdapter):
    """兼容老旧政务网站的 SSL 适配器：降低安全等级、允许旧版 TLS 与弱套件"""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.TLSv1
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        ctx.set_ciphers("DEFAULT@SECLEVEL=0")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)



def _download_scenic_csv() -> str:
    """调用接口并下载完整 CSV 文本（内部使用，不直接返回给大模型）。
    优先使用 curl_cffi 模拟浏览器 TLS 指纹，规避 OpenSSL 3.x 对该政务站点的
    `SSL: BAD_ECPOINT` 握手失败；若未安装 curl_cffi 则回退到 requests+LegacySSLAdapter。"""
    apply_url = "https://data.beijing.gov.cn/cms/web/bjdata/api/userApply.jsp"
    apply_params = {
        "id": "e9f52465cfd7478eb0768b9978623e2d85427",
        "key": BEIJING_USERKEY,
    }

    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        cffi_requests = None

    if cffi_requests is not None:
        # 方案A：curl_cffi（推荐，可绕过 BAD_ECPOINT）
        resp = cffi_requests.get(apply_url, params=apply_params, timeout=15,
                                 verify=False, impersonate="chrome")
    else:
        # 方案B：回退到 requests + 自定义 SSL 适配器
        session = requests.Session()
        session.mount("https://", LegacySSLAdapter())
        session.mount("http://", LegacySSLAdapter())
        resp = session.get(apply_url, params=apply_params, timeout=15, verify=False)

    try:
        info = resp.json()
    except ValueError:
        raise RuntimeError(f"接口未返回JSON，请检查 userKey。原始返回：{resp.text[:300]}")

    if str(info.get("code")) != "0":
        raise RuntimeError(f"接口返回异常：{info}")

    file_url = (info.get("result") or {}).get("address")
    if not file_url:
        raise RuntimeError(f"未获取到文件下载地址：{info}")

    if cffi_requests is not None:
        file_resp = cffi_requests.get(file_url, timeout=30, verify=False, impersonate="chrome")
        return file_resp.text
    else:
        file_resp = session.get(file_url, timeout=30, verify=False)
        file_resp.encoding = file_resp.apparent_encoding or "gbk"
        return file_resp.text


@tool
def get_beijing_scenic_data(scenic_name: str = "") -> str:
    """获取北京市重点景区饱和指数数据。可选参数 scenic_name 为景区名称（如“朝阳公园”），
    传入后只返回该景区相关的数据行，避免返回整份超大文件。"""
    try:
        content = _download_scenic_csv()
    except Exception as e:
        return f"数据获取失败：{e}"

    if not content or len(content.strip()) < 10:
        return "文件下载成功但内容为空。"

    # 解析 CSV，只保留目标景区相关行，并做长度上限保护
    name = (scenic_name or "").strip()
    try:
        reader = list(csv.reader(io.StringIO(content)))
    except Exception:
        reader = []

    if reader:
        header = reader[0]
        body = reader[1:]
        if name:
            matched = [r for r in body if any(name in cell for cell in r)]
        else:
            matched = body

        # 最多返回 100 行匹配数据，防止超长
        matched = matched[:100]
        lines = [",".join(header)] + [",".join(r) for r in matched]
        result = "\n".join(lines)
    else:
        result = content

    if not result.strip():
        return f"未找到景区“{name}”的数据。CSV 表头为：{reader[0] if reader else '无法解析'}"

    # 终极兜底：无论如何都不超过 8000 字符喂给模型
    MAX_CHARS = 8000
    if len(result) > MAX_CHARS:
        result = result[:MAX_CHARS] + f"\n...（内容过长已截断，共 {len(result)} 字符）"
    return result

@tool
def get_published_scenic_data(scenic_name: str = "") -> str:
    """获取已发布的北京市重点景区饱和指数数据（由本机定时任务发布到GitHub）。
    参数 scenic_name 为景区名称（如“长城”），只返回该景区相关数据行。
    当 get_beijing_scenic_data 因SSL错误或网络超时失败时，请使用本工具。"""
    try:
        resp = requests.get(PUBLISHED_DATA_URL, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        return f"发布数据获取失败：{e}"

    header = payload.get("header", [])
    body = payload.get("rows", [])
    name = (scenic_name or "").strip()
    matched = [r for r in body if any(name in cell for cell in r)] if name else body
    matched = matched[:100]
    if not matched:
        return f"发布数据中未找到景区“{name}”。数据更新时间：{payload.get('updated_at')}"

    lines = [",".join(header)] + [",".join(r) for r in matched]
    result = "\n".join(lines)
    if len(result) > 8000:
        result = result[:8000] + f"\n...（内容过长已截断，共 {len(result)} 字符）"
    return f"（数据发布时间：{payload.get('updated_at')}）\n{result}"

# 景区背景知识库工具
@tool
def get_scenic_knowledge(scenic_name: str) -> str:
    """调用高德地图API，查询北京景区的基础地理信息：地址、经纬度、开放时间、票价、评分、分类。
        当知识库knowledge_base.json缺少该景区信息时，调用本工具补充景区基础资料。
        参数 scenic_name：景区名称，会自动处理别名映射。
        """
    name = (scenic_name or "").strip()
    if not name:
        return "请传入景区名称"
    # 复用你已有的别名映射
    name = SCENIC_ALIASES.get(name, name)
    if not agent_gaodekey:
        return "未配置高德AMAP_KEY，无法调用高德接口"

    params = {
        "key": agent_gaodekey ,
        "keywords": name,
        "city": "北京"
    }
    try:
        resp = requests.get("https://restapi.amap.com/v3/place/text", params=params, timeout=12, verify=False)
        js = resp.json()
    except Exception as e:
        return f"高德接口请求失败：{str(e)}"

    if not js.get("pois"):
        return f"高德地图没有找到【{name}】的信息"
    poi = js["pois"][0]
    biz_ext = poi.get("biz_ext", {})
    info_text = f"""【高德地图景区信息-{poi['name']}】
    地址：{poi['address']}
    经纬度：{poi['lon']},{poi['lat']}
    分类：{poi['type']}
    开放时间：{biz_ext.get('opentime', '无')}
    门票价格：{biz_ext.get('cost', '无')}
    评分：{poi.get('rating', '无')}
    """
    return info_text

#h获取当前时间
def get_current_time():
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    weekday_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][current_time.weekday()]
    return f"当前时间是 {current_time_str}，今天是 {weekday_name}"

time_info = get_current_time()
print(time_info)
# 构建Agent
tools = [get_scenic_knowledge,get_beijing_scenic_data,get_published_scenic_data]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        f"当前时间是 {time_info}"
        "你是北京景区人流预测助手。收到用户提问的景区后，调用 get_beijing_scenic_data 和get_scenic_knowledge工具，"
        "如果get_beijing_scenic_data获取数据失败，不要放弃，请调用 get_published_scenic_data 工具。并说明"
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

def extract_forecast_json(text: str, marker: str = "DATA"):
    """从模型回答中提取指定标记(如DATA/WEEK)后面的预测JSON数组，返回 list 或 None"""
    if not text:
        return None
    # 优先找【marker:】标记后的内容
    match = re.search(rf"【{marker}:】\s*(\[.*?\])", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, list) and data else None
    except json.JSONDecodeError:
        return None

def render_forecast_chart(forecast: list, title: str = "人流饱和度预测", x_keys=("time", "时间", "date", "日期")):
    """把预测数据绘制成折线图并输出到 Streamlit"""
    df = pd.DataFrame(forecast)

    # 自动识别时间/日期列和饱和度列（兼容中英文字段名）
    time_col = next((c for c in df.columns if any(k in c.lower() or k in c for k in x_keys)), df.columns[0])
    value_col = next((c for c in df.columns if "饱和" in c or "人流" in c or "index" in c.lower()), df.columns[-1])

    # 时间转可排序的 datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    fig = px.line(
        df, x=time_col, y=value_col, markers=True,
        title=title
    )
    fig.update_layout(
        xaxis_title="时间", yaxis_title="人流饱和度(0-100)",
        height=420, yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit页面
st.title("北京景区人流查询Agent")
user_input = st.text_input("请输入查询的景点：")

if st.button("查询") and user_input:
    with st.spinner("思考中..."):
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        answer = result["messages"][-1].content

        # 分离展示：文字部分（去掉DATA/WEEK标记后面的JSON）
        text_part = re.split(r"【DATA:】|【WEEK:】", answer)[0]
        st.markdown(text_part)

        # 24小时预测折线图
        hourly = extract_forecast_json(answer, marker="DATA")
        if hourly:
            st.subheader("未来24小时人流预测")
            render_forecast_chart(hourly, title="未来24小时人流饱和度预测")
            with st.expander("查看24小时原始预测数据"):
                st.dataframe(pd.DataFrame(hourly))
        else:
            st.warning("未能解析出24小时预测数据。")

        # 七天预测折线图
        weekly = extract_forecast_json(answer, marker="WEEK")
        if weekly:
            st.subheader("未来七天人流预测")
            render_forecast_chart(weekly, title="未来七天每日人流饱和度预测")
            with st.expander("查看七天原始预测数据"):
                st.dataframe(pd.DataFrame(weekly))
        else:
            st.warning("未能解析出七天预测数据。")
