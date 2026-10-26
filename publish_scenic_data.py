"""本机运行：抓取北京景区数据 → 生成 scenic_data.json → git 推送到仓库供云端 agent 调用。
可配合 Windows 任务计划每天自动执行。"""
import io
import csv
import os
import json
import subprocess
from datetime import datetime

import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

BEIJING_USERKEY = os.environ.get("BEIJING_USERKEY")
APPLY_URL = "https://data.beijing.gov.cn/cms/web/bjdata/api/userApply.jsp"
API_ID = "e9f52465cfd7478eb0768b9978623e2d85427"
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenic_data.json")


def fetch_csv_text() -> str:
    try:
        from curl_cffi import requests as cffi_requests
        resp = cffi_requests.get(APPLY_URL, params={"id": API_ID, "key": BEIJING_USERKEY},
                                 timeout=15, verify=False, impersonate="chrome")
        info = resp.json()
        file_url = (info.get("result") or {}).get("address")
        if not file_url:
            raise RuntimeError(f"未获取到文件地址：{info}")
        return cffi_requests.get(file_url, timeout=30, verify=False, impersonate="chrome").text
    except ImportError:
        class LegacySSLAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                ctx = create_urllib3_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                ctx.minimum_version = ssl.TLSVersion.TLSv1
                ctx.maximum_version = ssl.TLSVersion.TLSv1_2
                ctx.set_ciphers("DEFAULT@SECLEVEL=0")
                kwargs["ssl_context"] = ctx
                return super().init_poolmanager(*args, **kwargs)
        session = requests.Session()
        session.mount("https://", LegacySSLAdapter())
        resp = session.get(APPLY_URL, params={"id": API_ID, "key": BEIJING_USERKEY},
                           timeout=15, verify=False)
        info = resp.json()
        file_url = (info.get("result") or {}).get("address")
        if not file_url:
            raise RuntimeError(f"未获取到文件地址：{info}")
        f = session.get(file_url, timeout=30, verify=False)
        f.encoding = f.apparent_encoding or "gbk"
        return f.text


def main():
    content = fetch_csv_text()
    rows = list(csv.reader(io.StringIO(content)))
    if not rows:
        raise RuntimeError("CSV 解析失败，内容为空")
    header, body = rows[0], rows[1:]
    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "header": header,
        "rows": body,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False)
    print(f"已生成 {OUT_FILE}，共 {len(body)} 行数据")

    # 自动提交并推送（要求脚本所在目录就是 git 仓库）
    repo_dir = os.path.dirname(OUT_FILE)
    for cmd in (["git", "add", "scenic_data.json"],
                ["git", "commit", "-m", f"data: refresh {data['updated_at']}"],
                ["git", "push"]):
        subprocess.run(cmd, cwd=repo_dir, check=False)
    print("已推送到 GitHub，云端 agent 可读取最新数据")


if __name__ == "__main__":
    main()