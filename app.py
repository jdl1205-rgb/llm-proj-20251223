# app.py
# ✅ CLI로 "입력 → 생성 → 채택 저장"을 실행하는 메인 파일입니다.
# ✅ API 키는 코드에 절대 넣지 않고 .env에서만 가져옵니다.

from __future__ import annotations

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

from prompts import (
    SYSTEM_PROMPT_V2,
    SHORTS_TYPE_LABEL,
    TONE_SUGGESTIONS,
    build_user_prompt,
)
from excel_store import append_adopted_row

# ✅ .env 파일에서 환경변수를 로드합니다.
#    (이 줄이 있어야 OPENAI_API_KEY가 환경변수로 올라옵니다.)
load_dotenv()

def ensure_api_key() -> None:
    """
    ✅ API 키가 환경변수에 있는지 확인만 합니다.
    ❌ 키 값을 출력하거나 로그로 남기지 않습니다.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다.\n"
            "1) 프로젝트 루트에 .env 파일을 만들고\n"
            "2) OPENAI_API_KEY=... 형태로 키를 넣어주세요.\n"
            "3) .env는 반드시 .gitignore에 포함하세요."
        )

def ask_int(prompt: str, min_v: int, max_v: int) -> int:
    """정수 입력을 안전하게 받는 함수"""
    while True:
        raw = input(prompt).strip()
        if not raw.isdigit():
            print(f"숫자만 입력해주세요. ({min_v}~{max_v})")
            continue
        v = int(raw)
        if v < min_v or v > max_v:
            print(f"범위를 벗어났습니다. ({min_v}~{max_v})")
            continue
        return v

def guess_item_count(topic_keyword: str) -> int | None:
    """
    '10가지', '8개', '7항목' 같은 표현에서 숫자를 뽑아옵니다.
    없으면 None 반환.
    """
    m = re.search(r"(\d+)\s*(가지|개|항목)", topic_keyword)
    if not m:
        return None
    return int(m.group(1))

def is_overloaded(shorts_type: int, video_length_sec: int, topic_keyword: str) -> bool:
    """
    ✅ 정보 과밀(편집 안내 문구) 조건 판단:
    - video_length_sec <= 12
    - shorts_type = 정보전달(3) 또는 팁/인사이트(6)
    - '10가지' 같은 항목 수 >= 6이 명시된 경우
    """
    if video_length_sec > 12:
        return False
    if shorts_type not in (3, 6):
        return False
    n = guess_item_count(topic_keyword)
    return (n is not None) and (n >= 6)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    ✅ OpenAI 호출 함수
    - 여기서도 API 키를 직접 다루지 않습니다.
    - OpenAI SDK가 환경변수(OPENAI_API_KEY)를 자동 사용합니다.
    """
    ensure_api_key()

    client = OpenAI()  # ✅ 키를 코드에 넣지 않음(환경변수 자동 사용)

    # 모델명은 교육 환경/계정에 따라 다를 수 있어요.
    # 에러 나면 "사용 가능한 모델명"을 너 환경에 맞게 바꿔주면 됩니다.
    resp = client.responses.create(
        model="gpt-4o-mini",
        instructions=system_prompt,
        input=user_prompt,
    )
    return resp.output_text

def extract_blocks(generated_text: str) -> tuple[str, str, str]:
    """
    ✅ 엑셀 저장용으로 블록을 대충 잘라 저장합니다.
    완벽 파싱보다 '안 깨지는 저장'이 목표입니다.
    """
    def get_block(tag: str, next_tags: list[str]) -> str:
        start = generated_text.find(tag)
        if start == -1:
            return ""
        start += len(tag)
        end = len(generated_text)
        for nt in next_tags:
            p = generated_text.find(nt, start)
            if p != -1:
                end = min(end, p)
        return generated_text[start:end].strip()

    title = get_block("[제목]", ["[타임라인]", "[비디오 프롬프트 - 한글]", "[Video Prompt - English]"])
    timeline = get_block("[타임라인]", ["[비디오 프롬프트 - 한글]", "[Video Prompt - English]"])
    scene_prompt_kr = get_block("[비디오 프롬프트 - 한글]", ["[Video Prompt - English]"])

    return title, scene_prompt_kr, timeline

def main() -> None:
    print("=== AI 쇼츠 생성기 (CLI) ===")

    # 1) 쇼츠 종류 선택
    print("\n쇼츠의 종류를 골라주세요. (숫자로 입력)")
    for i in range(1, 8):
        print(f"{i}. {SHORTS_TYPE_LABEL[i]}")
    shorts_type = ask_int("선택 (1~7): ", 1, 7)

    # 2) 영상 길이
    video_length_sec = ask_int("\n영상 길이(초)를 입력하세요 (예: 8/10/12): ", 3, 60)

    # 3) 톤(종류별 추천)
    suggestions = TONE_SUGGESTIONS.get(shorts_type, [])
    if suggestions:
        print("\n영상의 분위기 또는 톤을 입력하세요.")
        print(f"(추천: {', '.join(suggestions)})")
    else:
        print("\n영상의 분위기 또는 톤을 자유롭게 입력하세요.")
    tone = input("톤: ").strip()

    # 4) 캐릭터(사람/오브젝트)
    print("\n등장하는 캐릭터(또는 주된 대상/오브젝트)를 입력하세요.")
    print("(예: 아빠와 1살 아기 / 고양이와 집사 / 구슬 젤리와 칼)")
    character = input("캐릭터: ").strip()

    # 5) 토픽 키워드
    topic_keyword = input("\n토픽 키워드를 입력하세요: ").strip()

    # 6) 사용자 프롬프트 만들기
    user_prompt = build_user_prompt(
        shorts_type=shorts_type,
        video_length_sec=video_length_sec,
        tone=tone,
        character=character,
        topic_keyword=topic_keyword,
    )

    # (선택) 과밀이면 힌트 한 줄 추가 (시스템 프롬프트가 최종 통제)
    if is_overloaded(shorts_type, video_length_sec, topic_keyword):
        user_prompt += "\n(참고) 정보량이 많을 수 있으니 핵심 위주로 구성하세요.\n"

    # 7) 생성
    print("\n--- 생성 중 ---\n")
    try:
        result = call_llm(SYSTEM_PROMPT_V2, user_prompt)
    except Exception as e:
        print(f"[에러] LLM 호출 실패: {type(e).__name__}: {e}")
        return

    print(result)

    # 8) 채택 저장
    print("\n저장(채택)할까요?")
    print("1) 채택해서 엑셀 저장")
    print("2) 저장 안 함(종료)")
    choice = ask_int("선택 (1~2): ", 1, 2)

    if choice == 1:
        final_title, scene_prompt_kr, timeline = extract_blocks(result)

        append_adopted_row(
            video_length_sec=video_length_sec,
            tone=tone,
            character=character,
            topic_keyword=topic_keyword,
            final_title=final_title or "(제목 추출 실패)",
            scene_prompt=scene_prompt_kr or "(한글 비디오 프롬프트 추출 실패)",
            script=timeline or "(타임라인 추출 실패)",
        )
        print("\n✅ 엑셀(adopted_shorts.xlsx)에 저장했습니다.")
    else:
        print("\n✅ 저장하지 않고 종료합니다.")

if __name__ == "__main__":
    main()
