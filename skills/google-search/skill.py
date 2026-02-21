"""Google Search Skill — 使用 Gemini 2.5 Flash"""

import os
import sys
from pathlib import Path

# 載入 .env（如果存在）
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def google_search(query: str, language: str = "zh-TW") -> str:
    """
    用 Gemini 2.5 Flash 执行搜索。
    
    Args:
        query: 搜索问题
        language: 回复语言，默认繁体中文
    
    Returns:
        搜索摘要结果（纯文字）
    """
    if not HAS_GENAI:
        return "[错误] 请先安装: pip install google-generativeai"
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[错误] 未设定 GEMINI_API_KEY 环境变量"
    
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
            }
        )
        
        prompt = (
            f"请搜索以下问题并用{language}提供简洁准确的摘要，"
            f"只回传重点信息：\n\n{query}"
        )
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"[搜索失败] {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python skill.py \"搜索问题\"")
        sys.exit(1)
    
    query = sys.argv[1]
    result = google_search(query)
    print(result)
