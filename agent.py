import streamlit as st
import requests
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from datetime import datetime
import json
import plotly.express as px
import pandas as pd
import re

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



llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0
)

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
    if not agent_gaodekey:
        return "未配置高德AMAP_KEY，无法调用高德接口"

    params = {
        "key": agent_gaodekey,
        "keywords": name,
        "city": "北京",
        "region": "北京",
        "citylimit": "true",
        "extensions": "all",  # 必须 all 才返回 biz_ext/rating/opentime/cost 等深度信息
        "offset": "1",
        "page": "1",
    }
    try:
        resp = requests.get("https://restapi.amap.com/v3/place/text", params=params, timeout=12, verify=False)
        js = resp.json()
    except Exception as e:
        return f"高德接口请求失败：{str(e)}"

    if js.get("status") != "1":
        return f"高德接口返回异常：{js.get('info', js)}"
    if not js.get("pois"):
        return f"高德地图没有找到【{name}】的信息"

    poi = js["pois"][0]
    biz_ext = poi.get("biz_ext") or {}

    # 高德坐标在 location 字段，格式“经度,纬度”，没有独立的 lon/lat
    location = poi.get("location") or ""
    lon, lat = (location.split(",") + ["", ""])[:2] if location else ("", "")

    # address 可能为空列表，做兼容
    address = poi.get("address")
    if isinstance(address, list) or not address:
        address = "暂无"

    info_text = f"""【高德地图景区信息-{poi.get('name', name)}】
    地址：{address}
    经纬度：{lon},{lat}
    分类：{poi.get('type', '无')}
    开放时间：{biz_ext.get('opentime', '无') if isinstance(biz_ext, dict) else '无'}
    门票价格：{biz_ext.get('cost', '无') if isinstance(biz_ext, dict) else '无'}
    评分：{biz_ext.get('rating', '无') if isinstance(biz_ext, dict) else poi.get('rating', '无')}
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
tools = [get_scenic_knowledge,get_published_scenic_data]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        f"当前时间是 {time_info}"
        "你是北京景区人流预测助手。收到用户提问的景区后，调用 get_published_scenic_data 工具（景区拥挤度）和get_scenic_knowledge工具（景区基本情况），"
        "说明调用工具情况"
        "并把用户提到的景区名称作为 scenic_name 参数传入，只获取该景区相关数据。"
        "你获取的数据是以前的数据(时间不是当前时间)"
       """
        回答分两部分
        第一部分：基于返回的数据说明景区基本情况，预测【从当前时间开始】将来24小数内和七天内的人流情况，最后给出整体出行建议。
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
