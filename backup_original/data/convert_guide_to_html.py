"""
마크다운 가이드를 예쁜 HTML로 변환
브라우저에서 열고 Ctrl+P → PDF로 저장
"""

import markdown
import re

# 마크다운 파일 읽기
with open('data/GUIDE_파이썬_금융논문_완전초보가이드.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# 마크다운 → HTML 변환
html_body = markdown.markdown(
    md_content,
    extensions=['fenced_code', 'tables', 'toc']
)

# 코드 블록 스타일링 (python 하이라이팅)
html_body = re.sub(
    r'<code class="language-python">',
    '<code class="language-python" style="background:#f8f8f8; display:block; padding:10px; border-radius:5px; overflow-x:auto;">',
    html_body
)

# 전체 HTML
html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>파이썬 금융논문 완전 초보 가이드</title>
    <style>
        @media print {{
            body {{ font-size: 11pt; }}
            pre {{ page-break-inside: avoid; }}
            h1, h2, h3 {{ page-break-after: avoid; }}
        }}
        
        body {{
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            line-height: 1.8;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #333;
            background: #fff;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-top: 40px;
        }}
        
        h2 {{
            color: #2980b9;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 10px;
            margin-top: 35px;
        }}
        
        h3 {{
            color: #27ae60;
            margin-top: 25px;
        }}
        
        h4 {{
            color: #8e44ad;
            margin-top: 20px;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            line-height: 1.5;
            margin: 20px 0;
        }}
        
        pre code {{
            background: none;
            padding: 0;
            color: #f8f8f2;
            font-size: 0.85em;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        
        th {{
            background: #3498db;
            color: white;
        }}
        
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 15px 20px;
            background: #f8f9fa;
            color: #555;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #eee;
            margin: 40px 0;
        }}
        
        ul, ol {{
            padding-left: 25px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        /* 이모지 스타일 */
        .emoji {{
            font-size: 1.2em;
        }}
        
        /* 팁 박스 */
        p:has(strong:first-child) {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }}
        
        /* 링크 */
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        /* 목차 */
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .toc ul {{
            list-style: none;
            padding-left: 20px;
        }}
        
        .toc a {{
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    {html_body}
    
    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; color: #888; text-align: center;">
        <p>BACBB 프로젝트 기반 파이썬 금융논문 가이드</p>
    </footer>
</body>
</html>
'''

# HTML 파일 저장
with open('data/GUIDE_파이썬_금융논문_완전초보가이드.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✅ HTML 변환 완료!")
print("📄 파일: GUIDE_파이썬_금융논문_완전초보가이드.html")
print("")
print("📌 PDF로 저장하는 방법:")
print("   1. HTML 파일을 브라우저(Chrome)에서 열기")
print("   2. Ctrl + P (인쇄)")
print("   3. '대상'을 'PDF로 저장'으로 변경")
print("   4. '저장' 클릭")
