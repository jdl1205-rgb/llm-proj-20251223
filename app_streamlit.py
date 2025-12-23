# app_streamlit.py
from __future__ import annotations

import os
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIError

from prompts import (
    SYSTEM_PROMPT,
    SHORTS_TYPE_LABEL,
    TONE_SUGGESTIONS,
    build_user_prompt,
)

# =========================
# 환경 설정
# =========================
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="ShortMaker", page_icon="🎬", layout="wide")

if not api_key:
    st.title("🎬 ShortMaker")
    st.error(
        "OPENAI_API_KEY가 .env에 없습니다.\n\n"
        "프로젝트 폴더에 .env 파일을 만들고 아래처럼 넣어주세요:\n"
        "OPENAI_API_KEY=여기에_키"
    )
    st.stop()

client = OpenAI(api_key=api_key)

EXCEL_PATH = "shortmaker_results.xlsx"

REQUIRED_TAGS = [
    "[콘셉트]",
    "[제목 A]",
    "[제목 B]",
    "[타임라인]",
    "[비디오 프롬프트 - 한글]",
    "[AI 비디오 생성 프롬프트 - English (Sora)]",
]
FORBIDDEN_TAGS = ["[Video Prompt - English]"]

# 형식 오류 재시도
MAX_FORMAT_RETRIES = 2

# 네트워크/타임아웃 재시도
MAX_NETWORK_RETRIES = 3

# =========================
# LLM / 검증
# =========================
def call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    타임아웃/네트워크 오류를 대비해:
    - timeout 늘림
    - max_tokens 제한 (출력 과다 방지)
    """
    res = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=900,          # ✅ 너무 길어져서 늦어지는 것 방지
        timeout=60,              # ✅ 요청 타임아웃을 넉넉히(초)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (res.choices[0].message.content or "").strip()


def validate(text: str):
    missing = [t for t in REQUIRED_TAGS if t not in text]
    forbidden = [t for t in FORBIDDEN_TAGS if t in text]
    return missing, forbidden


def call_llm_with_network_retry(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    APITimeoutError 같은 네트워크 계열 에러는 자동 재시도.
    """
    last_err = None
    for attempt in range(1, MAX_NETWORK_RETRIES + 1):
        try:
            return call_llm(system_prompt, user_prompt, model)
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            last_err = e
            # ✅ 점점 기다렸다가 재시도 (1s, 2s, 4s)
            wait = 2 ** (attempt - 1)
            time.sleep(wait)

    # 여기까지 왔으면 끝내 실패
    raise last_err


def generate_with_retry(system_prompt: str, user_prompt: str, model: str):
    """
    1) 네트워크/타임아웃 재시도(call_llm_with_network_retry)
    2) 형식 오류 재시도(MAX_FORMAT_RETRIES)
    """
    output = ""
    missing, forbidden = [], []
    tries = 0

    fixup = (
        "\n\n[형식 재강조]\n"
        "- 반드시 OUTPUT FORMAT의 섹션 태그를 모두 포함하세요.\n"
        "- 금지된 섹션([Video Prompt - English])은 절대 포함하지 마세요.\n"
        "- 형식을 지키지 못하면 내용을 줄이더라도 형식을 우선하세요.\n"
    )

    for i in range(MAX_FORMAT_RETRIES + 1):
        tries = i + 1
        prompt_to_send = user_prompt if i == 0 else (user_prompt + fixup)

        output = call_llm_with_network_retry(system_prompt, prompt_to_send, model)
        missing, forbidden = validate(output)

        if not missing and not forbidden:
            break

    return output, missing, forbidden, tries


# =========================
# 파싱/표시 유틸
# =========================
def extract_sora_block(text: str) -> str:
    m = re.search(r"\[AI 비디오 생성 프롬프트 - English \(Sora\)\].*", text, re.DOTALL)
    return m.group(0).strip() if m else ""


def remove_sora_block_for_display(text: str) -> str:
    """
    결과창(st.code)에는 Sora 섹션을 제거해 중복을 없앤다.
    """
    pattern = r"\n*\[AI 비디오 생성 프롬프트 - English \(Sora\)\].*"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def extract_title_a(text: str) -> str:
    if "[제목 A]" not in text:
        return ""
    chunk = text.split("[제목 A]", 1)[1]
    for nt in ["[제목 B]", "[타임라인]", "[비디오 프롬프트 - 한글]", "[AI 비디오 생성 프롬프트 - English (Sora)]"]:
        if nt in chunk:
            chunk = chunk.split(nt, 1)[0]
            break
    return chunk.strip()


def append_to_excel(row: dict, excel_path: str = EXCEL_PATH) -> None:
    df_new = pd.DataFrame([row])

    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False, engine="openpyxl")


# =========================
# UI
# =========================
st.title("🎬 ShortMaker")
st.caption("AI 쇼츠 기획 + 타임라인 + Sora Shot 프롬프트 (유머 / ASMR)")

left, right = st.columns([1, 1])

if "last_output" not in st.session_state:
    st.session_state.last_output = ""
if "last_display" not in st.session_state:
    st.session_state.last_display = ""
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = {}
if "last_sora" not in st.session_state:
    st.session_state.last_sora = ""
if "last_title_a" not in st.session_state:
    st.session_state.last_title_a = ""

with left:
    st.subheader("입력")

    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)

    shorts_type = st.selectbox(
        "쇼츠 종류",
        options=[1, 2],
        format_func=lambda x: f"{x}. {SHORTS_TYPE_LABEL[x]}",
    )

    video_length_sec = st.number_input("영상 길이(초)", min_value=6, max_value=60, value=12, step=1)

    tone = st.text_input(
        "분위기 / 톤",
        value=TONE_SUGGESTIONS[shorts_type][0],
        help="추천: " + " / ".join(TONE_SUGGESTIONS[shorts_type]),
    )

    character_or_object = st.text_input("캐릭터 또는 오브젝트")
    topic_keyword = st.text_input("토픽 / 상황")

    generate = st.button("✨ 생성하기", use_container_width=True)

with right:
    st.subheader("결과")

    if generate:
        user_prompt = build_user_prompt(
            shorts_type=shorts_type,
            video_length_sec=int(video_length_sec),
            tone=tone,
            character_or_object=character_or_object,
            topic_keyword=topic_keyword,
        )

        try:
            with st.spinner("생성 중..."):
                output, missing, forbidden, tries = generate_with_retry(SYSTEM_PROMPT, user_prompt, model)

            st.session_state.last_output = output
            st.session_state.last_display = remove_sora_block_for_display(output)
            st.session_state.last_sora = extract_sora_block(output)
            st.session_state.last_title_a = extract_title_a(output)
            st.session_state.last_inputs = {
                "shorts_type": shorts_type,
                "video_length_sec": int(video_length_sec),
                "tone": tone,
                "character_or_object": character_or_object,
                "topic_keyword": topic_keyword,
            }

            if missing or forbidden:
                st.warning(
                    "형식 오류가 남아있습니다.\n\n"
                    f"- 시도: {tries}회\n"
                    f"- 누락: {missing}\n"
                    f"- 금지 포함: {forbidden}"
                )
            else:
                st.success(f"생성 완료 (형식 재시도 {tries}회 / 네트워크 재시도 최대 {MAX_NETWORK_RETRIES}회)")

        except APITimeoutError:
            st.error("OpenAI 요청이 시간 초과되었습니다(APITimeoutError). 네트워크 상태를 확인하고 다시 시도해 주세요.")
        except Exception as e:
            st.error(f"에러 발생: {type(e).__name__}: {e}")

    if not st.session_state.last_output:
        st.info("왼쪽에서 입력 후 '생성하기'를 눌러주세요.")
    else:
        st.markdown("### 생성 결과 (Sora 섹션 제외)")
        st.code(st.session_state.last_display, language="markdown")

        if st.session_state.last_sora:
            st.markdown("### Sora 복사용 (섹션 전체)")
            st.text_area("", st.session_state.last_sora, height=260)

        st.divider()
        adopt = st.button("✅ 이 결과 채택하고 엑셀에 저장", use_container_width=True)

        if adopt:
            inputs = st.session_state.last_inputs
            row = {
                "adopted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "shorts_type": SHORTS_TYPE_LABEL.get(inputs["shorts_type"], str(inputs["shorts_type"])),
                "video_length_sec": inputs["video_length_sec"],
                "tone": inputs["tone"],
                "character_or_object": inputs["character_or_object"],
                "topic_keyword": inputs["topic_keyword"],
                "title_a": st.session_state.last_title_a.strip(),
                "sora_prompt": st.session_state.last_sora.strip(),
                "output_full": st.session_state.last_output,
            }

            try:
                append_to_excel(row, EXCEL_PATH)
                st.success(f"엑셀 저장 완료: {EXCEL_PATH}")
            except Exception as e:
                st.error(f"엑셀 저장 실패: {type(e).__name__}: {e}")
