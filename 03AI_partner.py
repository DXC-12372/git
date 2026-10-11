import streamlit as st
import os
from openai import OpenAI
import json
from datetime import datetime
from streamlit.runtime import session_manager


#保存会话功能
def save_session():
    now_sessions = {
        "messages": st.session_state.messages,
        "partner_name": st.session_state.partner_name,
        "partner_nature": st.session_state.partner_nature,
        "session_name": st.session_state.session_name
    }#当前页面的对话
    if not os.path.exists("sessions"):
        os.makedirs("sessions")#创建目录,存放会话文件
    with open("sessions/%s.json" % st.session_state.session_name, "w", encoding="utf-8") as f:
        json.dump(now_sessions, f, ensure_ascii=False, indent=4)#把now_sessions变量的内容，以json格式写入f文件中

#加载会话列表
def load_sessions():
    session_list = []
    if os.path.exists("sessions"):
        file_list = os.listdir("sessions")#获取目录里的文件名
        for file in file_list:
            if file.endswith(".json"):
                session_list.append(file[:-5])
    session_list.sort(reverse=True)
    return session_list#存放会话名字的列表

#加载会话，读取文件中的会话内容到session变量，并加载到session_state中
def load_session(session_name):
    try:#尝试执行
        if  os.path.exists("sessions/%s.json" % session_name):
            with open("sessions/%s.json" % session_name, "r", encoding="utf-8") as f:
                session = json.load(f)#把文件f的内容加载成json格式，给到session变量中
                st.session_state.messages = session["messages"]
                st.session_state.partner_name = session["partner_name"]
                st.session_state.partner_nature = session["partner_nature"]
                st.session_state.session_name = session["session_name"]
    except Exception :
        st.error("加载会话出错")

#删除会话
def dele_session(session_name):
    try:
        if os.path.exists("sessions/%s.json" % session_name):
            os.remove("sessions/%s.json" % session_name)#删除文件
            if session_name == st.session_state.session_name:#如果删除的会话是当前会话
                st.session_state.messages = []#清空会话内容（新会话）
                st.session_state.session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")#更新会话时间（名称）
    except Exception :
        st.error("删除会话出错")

#设置页面配置项
st.set_page_config(
    page_title="AI智能伴侣",
    page_icon="😸",
    #布局
    layout="wide",#布局占满
    #侧边栏初始状态为展开
    initial_sidebar_state="expanded",
    #菜单设置
    menu_items={}
)

#系统提示词
system_prompt = """你叫%s，现在是用户的真实伴侣，请完全代入伴侣角色。：

                规则：
                    每次只回1条消息
                    1.
                    禁止任何场景或状态描述性文字
                    2.
                    匹配用户的语言
                    3.
                    回复简短，像微信聊天一样
                    4.
                    有需要的话可以用emoji表情
                    5.侣性格的方式对话
                    6.
                    要充分体现伴侣的性格特征
                    7.
                    回复的内容，要符合伴侣的性格特征
                伴侣性格:
                    %s
                    你必须严格遵守上述规则来回复用户。
                """

#初始化聊天信息
if 'messages' not in st.session_state: #st.session_state是一个字典.key访问value，储存信息
    st.session_state.messages = []#在session_state中创建.messages列表，后续一直储存在session_state.messages中


#初始化伴侣信息
if "partner_name" not in st.session_state:
    st.session_state.partner_name = "小甜甜"

#初始化性格
if "partner_nature" not in st.session_state:
    st.session_state.partner_nature = "活泼开朗"
#设定时间
if "session_name" not in st.session_state:
    st.session_state.session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#标题
st.title("AI智能伴侣")
st.logo("😸")

#显示聊天信息
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])#[key]取字典的值

#编辑侧边栏
with st.sidebar:
    st.subheader("AI控制面板")
    #新建会话按钮
    if st.button('新建会话',width='stretch',icon='🖋️'):#点击按钮返回True
        # 保存当前会话
        save_session()
        #如果当前会话有内容，则新建会话并保存
        if st.session_state.messages:
            st.session_state.session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            st.session_state.messages = []
            save_session()
            st.rerun()

    #历史会话
    st.text("历史会话")
    for session in load_sessions():#session是会话名字
        c1, c2 = st.columns([4,1])
        with c1:
            if st.button(session,width='stretch',icon='🗂️',key=f"load{session}",type = "primary" if session == st.session_state.session_name else "secondary"):
                save_session()
                load_session(session)
                st.rerun()

        with c2:
            if st.button("",width='stretch',icon='❌️',key=f"dele{session}"):
                dele_session(session)
                st.rerun()

    #分割线
    st.divider()

    #伴侣信息
    st.subheader("伴侣信息")
    partner_name = st.text_input("昵称",placeholder="请输入昵称",value=st.session_state.partner_name)
    if partner_name:
        st.session_state.partner_name = partner_name
    partner_nature = st.text_input("性格",placeholder="请输入性格",value=st.session_state.partner_nature)
    if partner_nature:
        st.session_state.partner_nature = partner_nature


#创建与AI大模型交互的客户端对象，使用HTTP协议，DEEPSEEK_API_KEY是环境变量名，值是API_KEY，需要import os
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
#openai库，调用大模型进行交互，client是客户端对象，用于发送请求和接收响应，base_url是大模型的地址
#OpenAI创建客户端的工具


#输入框
prompt = st.chat_input('Say something')
if prompt: #自动转化为布尔值
    st.chat_message("user").write(prompt)
    #保存用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("调用大模型，提示词：", prompt)
    #调用ai大模型
    response = client.chat.completions.create(
        model="deepseek-v4-pro",  # 调用的模型
        messages=[
            {"role": "system", "content": system_prompt%(st.session_state.partner_name, st.session_state.partner_nature)},
            *st.session_state.messages #解包*
        ],  # 参数
        stream=True
    )

    # 输出返回结果(流式输出)
    # print("AI助手回复：", response.choices[0].message.content)
    # st.chat_message("assistant").write(response.choices[0].message.content)

    #流式输出
    response_message = st.empty()#创建一个空的容器，可以不断更新
    full_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_message += chunk.choices[0].delta.content
            response_message.chat_message("assistant").write(full_message)#每次循环更新数据，不新建行
    #保存AI返回的结果
    st.session_state.messages.append({"role": "assistant", "content": full_message})
